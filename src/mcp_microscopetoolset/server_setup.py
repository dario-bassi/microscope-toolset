from typing import Any
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from src.local.prepare_code import prepare_code
import logging
import sys
import numpy as np
from io import BytesIO
from mcp.types import ImageContent
from PIL import Image

#  logger
logger = logging.getLogger("Server Setup")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)
fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(fh)



def create_mcp_server(
        database_agent,
        microscope_status,
        no_coding_agent,
        executor,
        logger_agent, 
        viewer,
        viewer_proxy=None
) -> FastMCP:
    # Server definition
    mcp = FastMCP(
        name="Microscope Toolset",
        host="127.0.0.1",
        port=5500,
        streamable_http_path="/mcp",
        log_level="INFO"
    )

    @mcp.tool(
        name="pymmcore_api_database",
        description="This tool is part of the feedback loop of the Microscope Toolset. It will return the relevant information"
                    "from the API database of pymmcore_plus. The relevant information will be searched by an hybrid method using"
                    "the reformulated query. The hybrid method will use the BM25 text matching and KNN search using embedding"
                    "vectors. Afterwards, a cross encoder will re-rank the result obtained to match only the most top 25 relevant"
                    "chunks of information."
    )
    def pymmcore_api_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:

        # reformulate user query
        reformulated_question = database_agent.rephrase_query(user_query)

        return database_agent.api_pymmcore_context(user_query, reformulated_question)
    @mcp.tool(
        name="micromanager_device_database",
        description="This tool is part of the feedback loop of the Microscope Toolset. It will return the relevant information"
                    "from the micromanager device. The relevant information will be searched by an hybrid method using"
                    "the reformulated query. The hybrid method will use the BM25 text matching and KNN search using embedding"
                    "vectors. Afterwards, a cross encoder will re-rank the result obtained to match only the most top 25 relevant"
                    "chunks of information."
    )
    def micromanager_device_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:

        # reformulate user query
        reformulated_question = database_agent.rephrase_query(user_query)

        return database_agent.devices_micromanager_context(user_query, reformulated_question)
    @mcp.tool(
        name="pdfs_publication_database",
        description="This tool is part of the feedback loop of the Microscope Toolset. It will return the relevant information"
                    "from a collection of scientific publications. The relevant information will be searched by an hybrid method using"
                    "the reformulated query. The hybrid method will use the BM25 text matching and KNN search using embedding"
                    "vectors. Afterwards, a cross encoder will re-rank the result obtained to match only the most top 25 relevant"
                    "chunks of information."
    )
    def pdfs_publication_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:

        # reformulate user query
        reformulated_question = database_agent.rephrase_query(user_query)

        return database_agent.pdf_publication_context(user_query, reformulated_question)

    @mcp.tool(
         name="reformulate_user_query",
         description="This tool is part of the feedback loop of the Microscope Toolset. It is used to rephrase the user question"
                     "that starts the feedback loop. The reformulated query will be used to search into different databases to retrieve"
                     "important information using text match with BM25 and embedding vectors."
     )
    def reformulate_user_query(
             user_question: str = Field(..., description="The user original question")
     ) -> dict[str, Any]:
         # add check that structured response is getting the correct answer
         return database_agent.rephrase_query(user_question)

    @mcp.tool(
        name="get_microscope_settings",
        description="This tool is part of the feedback loop of the Microscope Toolset. It has access to the settings of"
                    "a microscope. It returns a dictionary with the properties of the microscope, the current properties values "
                    "selected of each devices and the configuration groups saved into the microscope configuration file."
                    "This tool is useful to discover the properties and devices of the microscope."
    )
    def get_microscope_settings() -> dict[str, Any]:
        try:
            # Get Properties of the microscope
            logger.info("Getting microscope properties...")
            microscope_properties_response = microscope_status.get_properties()
            logger.info(f"Properties retrieved: {type(microscope_properties_response)}")
            
            # Get current settings
            logger.info("Getting microscope current status...")
            microscope_status_response = microscope_status.get_current_status()
            logger.info(f"Status retrieved: {type(microscope_status_response)}")
            
            # Get configuration settings
            logger.info("Getting microscope available configs...")
            config_settings = microscope_status.get_available_configs()
            logger.info(f"Configs retrieved: {type(config_settings)}")
            
            microscope_status_settings = {
                "properties_schema": microscope_properties_response,
                "current_properties_status": microscope_status_response,
                "configuration_groups_settings": config_settings
            }
            logger.info({
                "tool": "get_microscope_settings",
                "properties_schema": microscope_properties_response,
                "current_properties_status": microscope_status_response,
                "configuration_groups_settings": config_settings
            })
            return microscope_status_settings
        except Exception as e:
            logger.error(f"Error in get_microscope_settings: {e}", exc_info=True)
            return {
                "error": f"Failed to get microscope settings: {str(e)}",
                "properties_schema": {},
                "current_properties_status": {},
                "configuration_groups_settings": {}
            }

    @mcp.tool(
        name="answer_no_coding_query",
        description="This tool is part of the feedback loop of the Microscope Toolset.  It will flags if the main agent"
                    "will need to make an answer without any coding."
    )
    def answer_no_coding_query(
            user_query: str = Field(..., description="The user original query")
    ):
        return {
            "user_query": user_query,
            "no_coding_query": True
        }
    @mcp.tool(
        name="execute_python_code",
        description="""
        This tool is part of the feedback loop of the Microscope Toolset. It executes a given Python code 
        string and returns its output or any errors. If not already elaborated, you need to form a strategy and the code
        to run. Be aware of the precise constraints that these parameters have.
        """
    )
    def execute_python_code(
            user_query: str = Field(..., description="The user original query"),
            strategy: str = Field(..., description="""
            The strategy elaborated by the Main Agent.
            Given:
                - The original user query
                - The context (e.g., prior knowledge from the database or environment)
                - The microscope settings   
            
            
            Your main responsibility is to break the user's query into logical, sequenced steps, using available functions if possible.
            
            Build response
                - Based on the information, elaborate a strategy to answer the user query.
                - Break the query into smaller, logically ordered sub-tasks (if applicable).
                - Propose a concise, step-by-step strategy to address the user query.
            Response Style
                - Maintain a **scientific, concise, and unambiguous** communication style. Avoid redundant or non-technical phrasing.
            """),
            code: str = Field(..., description="""
            The code to be executed generated by the Main Agent.
            Given:
                - The original user query
                - The context (e.g., prior knowledge from the database or environment)
                - The microscope settings
                - The strategy of the Strategy Agent
            
            Your main responsibility is to generate Python code to answer the user's query using the strategy and all the available context information. Return raw text, don't format as markdown.
            
            Responsibilities
                - Use the strategy and the context to generate code that is:
                    - **Safe**: no security or hardware risks for the device or microscope.
                    - **Logical**: appropriate and functional.
                    - **Clear & Maintainable **: readable, cleanly structured.
                    - **Optimized**: efficient, minimal, and focused.
            
            Constrains
                - Use mmc (an instance of CMMCorePlus) to interact with the microscope.
                - Do **not** re-instantiate or reconfigure CMMCorePlus.
                - Only import essential, safe libraries.
                - **Avoid redundant narration** — just return the code in triple backticks.
                - **Print each result**, and if a value is None, print a human-readable message.
                - Include **minimal but meaningful comments** when needed.
                - We are using a GUI called napari-micromanager that is able to get a reference of the mmc object that you are using. Every images that you will produce will be shown in the GUI directly.
            """)
    ) -> dict[str, Any]:
        """
        Prepares and executes Python code using the Execute agent.
        Returns a dictionary with 'output' (the execution result) and 'error' (if any).
        """
        #code_string = code
        try:
            prepare_code_to_run = prepare_code(code)#code_string.strip("```")
            execution_output = executor.run_code(prepare_code_to_run)
            if "Error" in execution_output:
                logger.error({
                    "tool": "execute_python_code",
                    "user_query": user_query,
                    "strategy": strategy,
                    "code": code,
                    "error": execution_output
                })
                return {
                    "user_query": user_query,
                    "strategy": strategy,
                    "code": code,
                    "error": execution_output
                }
            else:
                logger.info({
                    "tool": "execute_python_code",
                    "user_query": user_query,
                    "strategy": strategy,
                    "code": code,
                    "is_final_output": True,
                    "output": execution_output
                })
                return {
                    "user_query": user_query,
                    "strategy": strategy,
                    "code": code,
                    "is_final_output": True,
                    "output": execution_output
                }
        except Exception as e:
            logger.error({
                "tool": "execute_python_code",
                "user_query": user_query,
                "strategy": strategy,
                "code": code,
                "error": f"Code preparation/execution failed: {e}"
            })
            return {
                "user_query": user_query,
                "strategy": strategy,
                "code": code,
                "error": f"Code preparation/execution failed: {e}"
            }

    @mcp.tool(
        name="save_result",
        description="Save the final result of the execution of the code",
    )
    def save_result(
            chat_summary: str = Field(..., description="This represents a summary of the current conversation that started when the user made a request and ended with the successfully run of the code."),
            feedback: bool = Field(..., description="This feedback describes if the final output was successful or not. If is not successfully will be False, True otherwise"),
            category: str = Field(..., description="This represents a one keyword that describe the category of the user request."),
            code: str = Field(..., description="The code to be executed generated by the SoftwareAgent"),
    ):
        """This function saves the final result of the execution of the code."""

        data = {
            "prompt": chat_summary,
            "output": code,
            "feedback": feedback,
            "category": category
        }
        try:

            # --new-- comment out for testing
            # add to database
            #database_agent.add_log(data)

            logger.info({
                "tool": "save_result",
            })
            return {
                "long_memory": "The previous result was successfully saved.",
            }

        except Exception as e:
            logger.error({e})
            return {
                "error_long_memory": "Unable to save the final result of the execution of the code.",
            }



    # @mcp.tool(
    #     name="save_result",
    #     description="This tool is part of the feedback loop of the Microscope Toolset. After showing the result to the user, "
    #                 "it will be asked to the user if the answer obtained was correct. We want to save into a database the "
    #                 "correct and the wrong answer to help you to answer the future user's questions. After you successfully "
    #                 "completed this, the user will likely ask you others questions or stop the server."
    # )
    # def save_result(
    #         user_query: str = Field(description="The user's response, typically 'correct' or 'wrong'.")
    # ):
    #     """
    #     Calls the LoggerAgent to save the output of the Agents into a database.
    #     Returns a json object with 'intent' (save) and a message.
    #     """
    #     # Get current data dict
    #     data_dict = microscope_session_object.get_data_dict()
    #     # evaluate user's answer
    #     if user_query == "correct":
    #         success = True
    #     elif user_query == "wrong":
    #         success = False
    #     else:
    #         loc_conversation = data_dict['conversation'] + [
    #             agent_message("Please specify if your query was answered or not using 'correct' or 'wrong'!")]
    #         microscope_session_object.update_data_dict(conversation=loc_conversation)
    #         return {"intent": 'error',
    #                 "message": "Please specify if your query was answered or not using 'correct' or 'wrong'!"}
    #     # prepare summary of the code
    #     summary_chat = logger_agent.prepare_summary(data_dict)
    #     if summary_chat.intent == "summary":
    #         data = {"prompt": summary_chat.message, "output": data_dict['code'], "feedback": success, "category": ""}
    #         # add into the db
    #         database_agent.add_log(data)
    #     # update the microscope session object
    #     microscope_session_object.reset_data_dict(old_output=data_dict['output'],
    #                                               old_microscope_status=data_dict['microscope_status'],
    #                                               old_microscope_properties=data_dict['microscope_properties'],
    #                                               old_microscope_presets=data_dict['configuration_presets'])
    #
    #     return {"intent": 'save', "message": "The previous result was added to the log database."}

    @mcp.tool(
        name="show_result",
        description="This tool is part of the feedback loop of the Microscope Toolset. Once the parameter 'is_final_output' "
                    "is changed to True, you need to show the user the final output."
    )
    def show_result(
            user_query: str = Field(..., description="The user original query"),
            strategy: str = Field(..., description="The strategy elaborated by the Strategy Agent"),
            code: str = Field(..., description="The code to be executed generated by the SoftwareAgent"),
            error: str = Field(..., description="The error when running the code"),
            is_final_output: bool = Field(..., description="Whether the code was executed successfully"),
            output: str = Field(..., description="The output of the code"),
    ):
        """
        Shows the user the final output
        """
        # # Get current data dict
        # data_dict = microscope_session_object.get_data_dict()
        data_dict = {
            "user_query": user_query,
            "strategy": strategy,
            "code": code,
            "error": error,
            "is_final_output": is_final_output,
            "output": output
        }
        # show the final output
        if data_dict['is_final_output']:
            final_output = data_dict['output']
            logger.info({
                "tool": "show_result",
                "output": final_output
            })
            return final_output
        else:
            message = "The final output was not reach yet!"
            logger.info({
                "tool": "show_result",
                "is_final_output": False,
                "message": message
            })
            return {
                "is_final_output": False,
                "message": message
            }
        
    # New Tool added
    # ------------------------------------------#
    # Napari Viewer
    # ------------------------------------------#
    @mcp.tool(
        name="viewer_session_information",
        description="Retrieve detailed information about the current napari viewer session. Returns metadata about the napari-micromanager viewer state including window size, available layers, camera position, and current display settings. Use this to understand the current state of the microscopy viewer before making changes."
    )
    def viewer_session_information():
        """
        Return information regarding the viewer session of napari micromanager
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('session_information')
        else:
            return viewer.session_information()
    
    # List of layer
    @mcp.tool(
        name="viewer_list_of_layers",
        description="Get a list of all layers currently loaded in the napari viewer with their properties (name, type, visibility, opacity, colormap). Use this to understand what image layers, label layers, and other data layers are present in the viewer and plan layer manipulation operations."
    )
    def viewer_list_of_layers():
        """
        Return a list of layers with all information
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('list_of_layers')
        else:
            return viewer.list_of_layers()
    
    # screenshot
    @mcp.tool(
        name="viewer_screenshot", 
        description="Capture a screenshot of the napari viewer's current state. This renders all visible layers and returns the image as an array. Set canvas_only=false to include GUI elements like scale bars and labels, or canvas_only=true to capture only the image data. Use this to visually inspect microscopy images or analyze image data for cell detection, segmentation, or other computer vision tasks.",
    )
    def viewer_screenshot(canvas_only: bool):
        """
        Return the ImageContent to pass the image data to the LLM
        """
        # Use viewer proxy if available to execute on main thread
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('screenshot', canvas_only=canvas_only)
        else:
            return viewer.screenshot(canvas_only)
    
    @mcp.tool(
        name="viewer_layer_screenshot", 
        description="Capture a screenshot of a specific layer from the napari viewer. Provide the exact layer name to isolate and render only that layer's data. Useful for examining individual microscopy channels, labeled regions, or segmentation masks without interference from other layers."
    )
    def viewer_layer_screenshot(layer_name: str):
        """
        Return the ImageContent of a specific layer to pass to the LLM
        """
        # Use viewer proxy if available to execute on main thread
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('layer_screenshot', layer_name=layer_name)
        else:
            return viewer.layer_screenshot(layer_name)
    

    # tools for open interact with napari viewer
    @mcp.tool(
        name="viewer_add_image",
        description="Load and display an image file in the napari viewer as a new layer. Provide the file path (supports common image formats), optional layer name, and visualization parameters like colormap (e.g., 'viridis', 'magma'), blending mode ('additive', 'translucent'), and channel_axis for multi-channel images. Use this to add microscopy images, fluorescence channels, or processed image data to the viewer for analysis and visualization."
    )
    def viewer_add_image(
        path: str,
        name: str | None = None,
        colormap: str | None = None,
        blending: str | None = None,
        channel_axis: int | str | None = None
    ):
        """
        Add an image layer from a file path
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_image', path=path, name=name, colormap=colormap, blending=blending, channel_axis=channel_axis)
        else:
            return viewer.add_image(path, name, colormap, blending, channel_axis)
    
    @mcp.tool(
        name="viewer_add_labels",
        description="Add a segmentation/labels layer from an image file containing labeled regions. Each unique integer value in the image represents a distinct region (e.g., individual cells, nucleus, organelles). Provide the file path to the labels image and an optional layer name. Use this to display cell detection results, segmentation masks, or any labeled image analysis results in the napari viewer with automatic color mapping for easy visualization of individual regions."
    )
    def viewer_add_labels(
        path: str, 
        name: str | None = None
    ):
        """
        Add a labels layer from a file
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_labels', path=path, name=name)
        else:
            return viewer.add_labels(path, name)
        
    @mcp.tool(
        name="viewer_add_points",
        description="Add a points layer to the napari viewer for marking locations of interest. Provide a list of coordinate pairs (2D) or triples (3D) representing point positions in pixel/voxel space. Optionally set the layer name and point size for visualization. Use this to annotate cell locations, mark regions of interest, indicate measurement points, or overlay coordinate data on microscopy images."
    )
    def viewer_add_points(
        points: list[list[float]], 
        name: str | None = None,
        size: int | str = 10
    ):
        """
        Add a points layer
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_points', points=points, name=name, size=size)
        else:
            return viewer.add_points(points, name, size)
        
    
    @mcp.tool(
        name="viewer_remove_layer",
        description="Remove a layer from the napari viewer by its exact name. Use this to clean up the viewer workspace by deleting intermediate processing results, redundant layers, or layers that are no longer needed for analysis. Check the current layers with viewer_list_of_layers before removing."
    )
    def viewer_remove_layer(name: str):
        """
        Remove an existince layer
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('remove_layer', name=name)
        else:
            return viewer.remove_layer(name)
    

    @mcp.tool(
        name="viewer_set_layer_properties",
        description="Modify visual properties of a layer in the napari viewer. Adjust visibility (True/False), opacity (0-1, where 0 is transparent), colormap ('viridis', 'magma', 'red', etc.), blending mode ('additive', 'translucent'), contrast limits for brightness/contrast adjustment, gamma for exposure, and optionally rename the layer. Use this to improve visualization, highlight specific features, or enhance contrast for better image analysis."
    )
    def viewer_set_layer_properties(
        name: str, 
        visible: bool | None = None,
        opacity: float | None = None,
        colormap: str | None = None,
        blending: str | None = None,
        contrast_limits: list[float] | None = None,
        gamma: float | str | None = None,
        new_name: str | None = None
    ):
        """
        Set common properties on a layer name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_layer_properties', name=name, visible=visible, opacity=opacity, colormap=colormap, blending=blending, contrast_limits=contrast_limits, gamma=gamma, new_name=new_name)
        else:
            return viewer.set_layer_properties(name,visible,opacity,colormap,blending,contrast_limits,gamma,new_name)
    @mcp.tool(
        name="viewer_reorder_layer",
        description="Change the stacking order (z-order) of layers in the napari viewer. Specify the layer name and either an absolute index, or position it before/after another named layer. Use this to control which layers appear on top when layers overlap, which affects visibility in multi-layer microscopy visualizations where layer stacking order matters for interpretation."
    )
    def viewer_reorder_layer(
        name: str,
        index: int | str | None = None,
        before: str | None = None,
        after: str | None = None
    ):
        """
        Reorder a layer by name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('reorder_layer', name=name, index=index, before=before, after=after)
        else:
            return viewer.reorder_layer(name, index, before, after)
    
    @mcp.tool(
        name="viewer_set_active_layer",
        description="Select/activate a specific layer in the napari viewer by name. The active layer is highlighted in the layers panel and operations like drawing, annotation, or selection tools apply to this layer. Use this when you need to work with a specific layer or prepare a layer for editing."
    )
    def viewer_set_active_layer(
        name: str
    ):
        """
        Set the selected/active layer by name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_active_layer', name=name)
        else:
            return viewer.set_active_layer(name)
    
    @mcp.tool(
        name="viewer_reset_view",
        description="Reset the camera view to fit all visible data layers optimally in the viewer window. This adjusts zoom and pan to show the entire image extent. Use this to get a complete overview of your data after zooming into specific regions, or to standardize the view between different analyses."
    )
    def viewer_reset_view():
        """
        Reset the camera view to fit the data
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('reset_view')
        else:
            return viewer.reset_view()
    
    @mcp.tool(
        name="viewer_set_camera",
        description="Control the camera viewing parameters in the napari viewer. Set the center position to pan to a specific region, zoom level to magnify (larger = more zoom), and angle for 3D rotation (if working in 3D mode). Use this to navigate to regions of interest, zoom in on details, or create consistent viewing angles for image documentation."
    )
    def viewer_set_camera(
        center: list[float] | None = None,
        zoom: float | str | None = None,
        angle: float | str | None = None
    ):
        """
        Set the camera properties: center, zoom, angle
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_camera', center=center, zoom=zoom, angle=angle)
        else:
            return viewer.set_camera(center, zoom, angle)
    
    @mcp.tool(
        name="viewer_set_ndisplay", 
        description="Switch the napari viewer between 2D and 3D display modes. Set ndisplay=2 for standard 2D microscopy slice viewing, or ndisplay=3 for 3D volumetric visualization when working with Z-stack or 3D image data. Use this to toggle between 2D slice inspection and 3D volume rendering."
    )
    def viewer_set_ndisplay(
        ndisplay: int | str
    ):
        """
        Set number of displayed dimension (2 or 3)
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('_set_ndisplay', ndisplay=ndisplay)
        else:
            return viewer._set_ndisplay(ndisplay)
    
    @mcp.tool(
        name="viewer_set_dims_current_step",
        description="Navigate through a specific dimension (axis) of multi-dimensional image data. Provide the axis name/index (e.g., 'Z' for Z-stack depth, 0, 1, 2, etc.) and the step value. Use this to browse through Z-slices in a Z-stack, time frames in a time-lapse, or channels in multi-channel images. This is equivalent to moving the slider for that dimension."
    )
    def viewer_set_dims_current_step(
        axis: int | str, 
        value: int | str
    ):
        """
        Set the current step (slider position for a specific axis)
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_dims_current_step', axis=axis, value=value)
        else:
            return viewer.set_dims_current_step(axis, value)
    
    @mcp.tool(
        name="viewer_set_grid",
        description="Toggle the display of a pixel grid overlay in the napari viewer. Set enabled=true to show the grid (useful for precise pixel-level measurements and alignment), or enabled=false to hide it for a cleaner view. Use this to switch between detailed pixel-level work and overview visualization modes."
    )
    def set_grid(enabled: bool | str = True):
        """
        Enable or disable grid view
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_grid', enabled=enabled)
        else:
            return viewer.set_grid(enabled)
    
    # TODO: add timelapse_screenshot later
    


    return mcp


def run_server(mcp: FastMCP) -> None:
    """
    Start the MCP Microscope Toolset server.
    """
    logger.info("MCP Microscope Toolset server started using streamable-http.")
    mcp.run(transport="streamable-http")