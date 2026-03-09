from mcp.types import ImageContent
import imageio.v3 as iio
import numpy as np
import logging
import sys
import re
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
from napari import Viewer
from typing import Any
import contextlib
import tifffile
from cellpose import models
from cellpose.io import imread, imsave
import torch


#  logger
logger = logging.getLogger("Viewer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


class NapariViewerMC:

    def __init__(self, viewer: Viewer) -> None:
        self._viewer = viewer


    def session_information(self):
        """
        Returning information regarding the viewer session of napari micromanager. 
        """
        if self._viewer is None:
            return {
                "status": "error",
                "message": "No Viewer exist! Somenthing went wrong."
            }
        
        # Get information about the Viewer
        viewer_infos = {
            "title": self._viewer.title, 
            "n_layers": len(self._viewer.layers), # number of layers (layers is a List)
            "layers_names": [layer.name for layer in self._viewer.layers], # get the name of each layers
            "selected_layers": [layer.name for layer in self._viewer.layers.selection],
            "current_step": dict(enumerate(self._viewer.dims.current_step)) if hasattr(self._viewer.dims, "current_step") else {}, # current step for each dimension
            "ndisplay": self._viewer.dims.ndisplay, # number of displayed dimensions
            "camera_center": list(self._viewer.camera.center), #Center of rotation for the camera. In 2D viewing the last two values are used.
            "camera_zoom": float(self._viewer.camera.zoom), # Scale from canvas pixels to world pixels.
            "camera_angles": list(self._viewer.camera.angles) if self._viewer.camera.angles else [],
            "grid_enabled": self._viewer.grid.enabled
        }

        # Layers information
        layers_details = []
        for layer in self._viewer.layers:
            layers_detail = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "visible": getattr(layer, "visible", True),
                "opacity": getattr(layer, "opacity", 1.0),
                "blending": getattr(layer, "blending", ""),
                "data_shape": list(layer.data.shape),
                "data_dtype": str(layer.data.dtype)
            }

            # Layers specific propertied
            if hasattr(layer, "colormap"):
                layers_detail["colormap"] = getattr(layer.colormap, "name", str(layer.colormap))

            if hasattr(layer, "contrast_limits"):
                try:
                    cl = layer.contrast_limits
                    layers_detail["contrast_limits"] = [float(cl[0]), float(cl[1])]
                except Exception:
                    pass

            if hasattr(layer, "gamma"):
                layers_detail["gamma"] = float(getattr(layer, "gamma", 1.0))

            layers_details.append(layers_detail)
        return {
            "viewer": viewer_infos,
            "layers": layers_details
        }
    
    def list_of_layers(self):
        """
        Return a list of all the layers
        """

        result: list[dict[str, Any]] = []

        for lyr in self._viewer.layers:
            entry = {
                "name": lyr.name,
                "type": lyr.__class__.__name__,
                "visible": getattr(lyr, "visible", True),
                "opacity": getattr(lyr, "opacity", 1.0),
                "blending": getattr(lyr, "blending", "")
            }

            if hasattr(lyr, "colormap") and getattr(lyr, "colormap", "") != "":
                entry["colormap"] = getattr(lyr.colormap, "name", "") or str(lyr.colormap) # to check
            

            if hasattr(lyr, "contrast_limits") and getattr(lyr, "contrast_limits", None) is not None:
                try:
                    cl = list(lyr.contrast_limits)
                    entry["contrast_limits"] = [float(cl[0]), float(cl[1])]
                except Exception:
                    pass

            
            result.append(entry)

        return {"layers": result}
    
    def screenshot(self, canvas_only: bool):
        """
        Return a screenshot from the current viewer.

        If canvas is True, it will return only the canva areas, otherwise
        it will return a screenshot from the all window of napari micromanager.
        """
        img_arr = self._viewer.screenshot(canvas_only=canvas_only)
        
        return self._transform_array_to_image_content(img_arr)
    
    def layer_screenshot(self, layer_name: str):
        """
        Return a screenshot of a single layer rendered by napari.

        Uses napari's own rendering pipeline: hides all layers except the target,
        takes a canvas screenshot, then restores visibility. This correctly handles
        Labels layers (color-coded), colormaps, contrast limits, and current slice.
        """
        if layer_name not in [layer.name for layer in self._viewer.layers]:
            return {
                "status": "error",
                "message": "The layer_name doesn't exist! Please check the name."
            }

        # Save visibility state of all layers
        visibility_state = {layer.name: layer.visible for layer in self._viewer.layers}

        try:
            # Hide all layers except the target
            for layer in self._viewer.layers:
                layer.visible = (layer.name == layer_name)

            # Use napari's own renderer — always produces RGBA uint8
            img_arr = self._viewer.screenshot(canvas_only=True)
        finally:
            # Restore all layers' visibility
            for layer in self._viewer.layers:
                if layer.name in visibility_state:
                    layer.visible = visibility_state[layer.name]

        return self._transform_array_to_image_content(img_arr)
    
    def get_layer_data(self, layer_name: str, save_path: str | None = None):
        """
        Export raw numpy data from a napari layer to a TIFF file on disk.

        This is the safe data bridge between the viewer (main thread) and
        execute_python_code (daemon thread). Returns metadata about the
        exported layer.

        Supports Image and Labels layers. Returns error for unsupported types.
        """
        if layer_name not in [layer.name for layer in self._viewer.layers]:
            return {
                "status": "error",
                "message": f"Layer '{layer_name}' not found. Use viewer_list_of_layers to see available layers."
            }

        layer = self._viewer.layers[layer_name]
        layer_type = layer.__class__.__name__

        if layer_type not in ("Image", "Labels"):
            return {
                "status": "error",
                "message": f"Unsupported layer type '{layer_type}'. Only Image and Labels layers are supported."
            }

        data = np.asarray(layer.data)

        # Build default save path from sanitized layer name
        if save_path is None:
            sanitized = re.sub(r'[^\w\-.]', '_', layer_name)
            save_path = f"/tmp/{sanitized}.tif"

        try:
            tifffile.imwrite(save_path, data)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to write TIFF: {e}"
            }

        result = {
            "status": "success",
            "path": save_path,
            "layer_type": layer_type,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }

        if layer_type == "Labels":
            unique = np.unique(data)
            result["num_labels"] = int(len(unique) - 1) if 0 in unique else int(len(unique))
            result["unique_labels"] = unique[:50].tolist()
        elif layer_type == "Image":
            result["min"] = float(np.min(data))
            result["max"] = float(np.max(data))
            result["mean"] = float(np.mean(data))

        return result

    def _transform_array_to_image_content(self, arr: np.ndarray) -> dict[str, Any]:#ImageContent
        """Helper function to transfor the array in a ImageContent"""

        # Ensure array is a NumPy array with proper dtype
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        
        # Convert to uint8 if needed, allowing copy when necessary
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        enc = buf.getvalue()

        base64_img = base64.b64encode(enc).decode("utf-8")

        return ImageContent(
            type="image",
            data=base64_img,
            mimeType="image/png"
        )
    
    def add_image(
            self,
            path: str | None = None,
            img_data: np.ndarray | list[np.ndarray] | None = None,
            name: str | None = None,
            colormap: str | None = None,
            blending: str | None = None,
            channel_axis: int | str | None = None
            ):
        """
        Add an image to a layer in the current viewer session
        """
        try:
            
            if path is not None:
                img_data = iio.imread(path)
            else:
                img_data = img_data

            # add image
            layer = self._viewer.add_image(
                img_data,
                name=name,
                colormap=colormap,
                blending=blending,
                channel_axis=channel_axis
            )

            return {
                "status": "success",
                "name": layer.name,
                "shape": list(np.shape(img_data))
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add image from {path}: {e}"
            }
        
    def add_labels(
            self, 
            path: str | None = None,
            img_data: np.ndarray | None = None, 
            name: str | None = None
            ):
        """
        Add an label layer from a file
        """
        try:
            if path is not None and img_data is None:
                p = Path(path).expanduser().resolve(strict=False)
                img = iio.imread(str(p))
                layer = self._viewer.add_labels(img, name=name)
                return {
                "status": "success",
                "name": layer.name,
                "hsape": list(np.shape(img))
            }
            elif path is None and img_data is not None:
                layer = self._viewer.add_labels(img_data, name=name)

                return {
                    "status": "success",
                    "name": layer.name,
                    "hsape": list(np.shape(img_data))
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add labels from {path}: {e}"
            }
        

    def add_points(
            self,
            points: list[list[float]], 
            name: str | None = None,
            size: int | str = 10
            ):
        """
        Add a points layers
        """
        try:

            arr = np.asarray(points, dtype=float)
            layer = self._viewer.add_points(arr, name=name, size=int(size))

            return {
                "status": "success",
                "name": layer.name,
                "n_points": int(arr.shape[0])
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add points layer: {e}"
            }


    def remove_layer(self, name: str):
        """
        Docstring for remove_layer
        
        :param self: Description
        :param name: Description
        :type name: str

        Remove an existe layer
        """
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
            return {
                "status": "success", 
                "message": f"The layer {name} was successfully removed."
            }
        
        return {
            "status": "error",
            "message": f"The layer {name} doesn't exists."
        }


    def set_layer_properties(
            self,
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
        Docstring for set_layer_properties
        
        :param self: Description
        :param name: Description
        :type name: str
        :param visible: Description
        :type visible: bool | None
        :param opacity: Description
        :type opacity: float | None
        :param colormap: Description
        :type colormap: str | None
        :param blending: Description
        :type blending: str | None
        :param contrast_limits: Description
        :type contrast_limits: list[float] | None
        :param gamma: Description
        :type gamma: float | str | None
        :param new_name: Description
        :type new_name: str | None

        set a layer with specific properties
        """
        if name not in self._viewer.layers:
            return {
                "status": "error",
                "message": f"The layer {name} doesn't exist."
            }
        selected_layers = self._viewer.layers[name]

        if visible is not None and hasattr(selected_layers, "visible"):
            selected_layers.visible = visible
        if opacity is not None and hasattr(selected_layers, "opacity"):
            selected_layers.opacity = float(opacity)
        if colormap is not None and hasattr(selected_layers, "colormap"):
            selected_layers.colormap = colormap
        if blending is not None and hasattr(selected_layers, "blending"):
            selected_layers.blending = blending
        if contrast_limits is not None and hasattr(selected_layers, "contrast_limits"):
            with contextlib.suppress(Exception):
                selected_layers.contrast_limits = [
                    float(contrast_limits[0]),
                    float(contrast_limits[1])
                ]
        if gamma is not None and hasattr(selected_layers, "gamma"):
            selected_layers.gamma = float(gamma)
        if new_name is not None:
            selected_layers.name = new_name

        return {
            "status": "success",
            "message": f"Common properties were set for layer {new_name}"
        }
    

    def reorder_layer(
            self,
            name: str,
            index: int | str | None = None,
            before: str | None = None,
            after: str | None = None
    ):
        """
        Docstring for reorder_layer
        
        :param self: Description
        :param name: Description
        :type name: str
        :param index: Description
        :type index: int | str | None
        :param before: Description
        :type before: str | None
        :param after: Description
        :type after: str | None

        Reorder layer from an image
        """
        if name not in self._viewer.layers:
            return {
                "status": "error",
                "message": f"The layer {name} doesn't exist."
            }
        if sum(x is not None for x in (index, before, after)) != 1:
            return {
                "status": "error",
                "message": "Provide exactly one of index, or before or after"
            }
        
        cur = self._viewer.layers.index(name)
        target = cur
        if index is not None:
            target = max(0, min(int(index), len(self._viewer.layers) - 1))
        elif before is not None:
            if before not in self._viewer.layers:
                return {
                    "status": "error",
                    "message": f"The layer {before} doesn't exist."
                }
            target = self._viewer.layers.index(before)
        elif after is not None:
            if after not in self._viewer.layers:
                return {
                    "status": "error",
                    "message": f"The layer {after} doesn't exist."
                }
            target = self._viewer.layers.index(after)
        
        if target != cur:
            self._viewer.layers.move(cur, target)

        return {
            "status": "successfull",
            "message":f"The layer {name} was moved at the new index {self._viewer.layers.index(name)}"

        }
    
    def set_active_layer(self, name: str):
        """
        Docstring for set_active_layer
        
        :param self: Description
        :param name: Description
        :type name: str

        Set the new activae layer from the session
        """
        if name not in self._viewer.layers:
            return {
                "status": "error",
                "message": f"The layer {name} doesn't exist."
            }
        self._viewer.layers.selection = {self._viewer.layers[name]}

        return {
            "status": "success",
            "message": f"The new activate layer {name} was set."
        }
    
    def reset_view(self):
        """
        Docstring for reset_view
        
        :param self: Description

        Reset the view to contain all the data
        """
        self._viewer.reset_view()

        return {
            "status": "success", 
            "message": "The camera view was reset."
        }
    
    def set_camera(
            self,
            center: list[float] | None = None,
            zoom: float | str | None = None,
            angle: float | str | None = None
    ):
        """
        Docstring for set_camera
        
        :param self: Description
        :param center: Description
        :type center: list[float] | None
        :param zoom: Description
        :type zoom: float | str | None
        :param angle: Description
        :type angle: float | str | None

        Set the new camera
        """
        if center is not None:
            self._viewer.camera.center = list(map(float, center))
        if zoom is not None:
            self._viewer.camera.zoom = float(zoom)
        if angle is not None:
            self._viewer.camera.angles = (float(angle),)

        return {
            "status": "success",
            "center": list(map(float, self._viewer.camera.center)),
            "zoom": float(self._viewer.camera.zoom)
        }
    
    def set_ndisplay(self, ndisplay: int | str):
        """
        Docstring for set_ndisplay
        
        :param self: Description
        :param ndisplay: Description
        :type ndisplay: int | str

        Set the number of displayed dimension (2 or 3)
        """
        self._viewer.dims.ndisplay = int(ndisplay)

        return {
            "status": "success",
            "message": f"The number of displayed dimension was set to {ndisplay}"
        }

    def set_dims_current_step(self, axis: int | str, value: int | str):
        """
        Docstring for set_dims_current_step
        
        :param self: Description
        :param axis: Description
        :type axis: int | str
        :param value: Description
        :type value: int | str

        Set the current step (slider position for a specific axis)
        """
        self._viewer.dims.set_current_step(int(axis), int(value))

        return {
            "status": "success", 
            "message": f"For the axis {axis} was set {value}"
        }
    
    def set_grid(self, enabled: bool = True): # bool | str
        """
        Docstring for set_grid
        
        :param self: Description
        :param enabled: Description
        :type enabled: bool | str

        Enable or Disable the grid view
        """
        self._viewer.grid.enabled = enabled

        return {
            "status": "success",
            "message": f"The grid view was set to {enabled}"
        }
    

    def add_tracks(self, 
                   track_data: np.ndarray,
                   features: dict[str, Any] | None = None, 
                   tail_width: float | None = None, 
                   tail_length: float | None = None):
        """
        This function add a tracks layer to layer list.

        Parameters:
            track_data: NxD+1 NumPy Array or list containig the coordinates of N vertices with a 
                track ID and coordinats in D dimensions. The ordering of these dimensions is the same 
                as the ordering of the dimensions for image layers. This array is always accessible through the 
                layer.data property and will grow or shrink as new tracks are either added or deleted.
                The Tracks layer assumes the first column is the track_id, the second column is the time axis, 
                and columns 3-5 are Z, Y, and X, respectively. Other feature can be added in other coloumns. 
                Each row is one vertex in a track. All vertices with the same track_id are joined into a single track.

            features: Features table where each row corresponds to a point and each column is a feature.

            tail_width: Float value representing the width of the track tails in pixels.

            tail_length: Float value representing the length of the positive (backward in time) tails in units of time.

        """

        try:
            kwargs = {"data": np.asarray(track_data)}
            if features is not None:
                kwargs["features"] = features
            if tail_width is not None:
                kwargs["tail_width"] = int(tail_width)
            if tail_length is not None:
                kwargs["tail_length"] = int(tail_length)
            self._viewer.add_tracks(**kwargs)

            return {
                "status": "success",
                "message": "The tracks data was successfully added."
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to add tracks: {e}"
            }
        
    
    def segment_image(self,
                       path: str | None = None,
                       img: np.ndarray | list[np.ndarray] | None = None,
                       mask_name: str | None = None,
                       batch_size: int = 8,
                       resample: bool = True,
                       channels_axis: int | None = None, 
                       z_axis: int | None = None,
                       normalize: bool = True,
                       rescale: float | None = None,
                       diameter: float | list[float] | None = None,
                       flow_threshold: float = 0.4,
                       cellprob_threshold: float = 0.0,
                       augment: bool = False):
        """
        This function segment a cell using cellpose
        """
        try:
            # chek if GPU is available
            if torch.cuda.is_available():
                gpu_available = True
            else:
                gpu_available = False

            # select cellpose model
            model = models.CellposeModel(gpu=gpu_available)

            # read an image
            if path is not None:
                img_array = imread(path)
            else:
                img_array = img
            
            # check if the image is a 2d,3d,4d image
            # define CHANNELS to run segementation on
            # grayscale=0, R=1, G=2, B=3
            # channels = [cytoplasm, nucleus]
            # if NUCLEUS channel does not exist, set the second channel to 0
            # channels = [0,0]
            # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
            # channels = [0,0] # IF YOU HAVE GRAYSCALE
            # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
            # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
            # OR if you have different types of channels in each image
            # channels = [[2,3], [0,0], [0,0]]

            # if you have a nuclear channel, you can use the nuclei restore model on the nuclear channel with
            # model = denoise.CellposeDenoiseModel(..., chan2_restore=True)

            masks, flows, styles = model.eval(img_array, 
                                              batch_size=batch_size, 
                                              resample=resample, 
                                              channel_axis=channels_axis,
                                              z_axis=z_axis,
                                              normalize=normalize, 
                                              rescale=rescale,
                                              diameter=diameter,
                                              flow_threshold=flow_threshold,
                                              cellprob_threshold=cellprob_threshold,
                                              augment=augment)
            
            # save the masks
            if mask_name is not None:
                name = mask_name
            else:
                name = "Segmentation"
            
            # checks if img is a np.ndarray or list of arrays
            if isinstance(masks, np.ndarray):
                # create tmp folder
                default_path = f"/tmp/{name}_mask.tif"
                # save mask in .tif file
                imsave(default_path, masks)

                # add labels in napari gui
                self._viewer.add_labels(masks, name=name)

                return {
                    "status": "success",
                    "mask_file_path": default_path,
                    "masks": masks,
                    "number_of_masks": len(masks)
                }
            else: # list of arrays
                for i,img in enumerate(masks):
                    default_path = f"/tmp/{name}_mask_{i}.tif"
                    # save mask in .tif file
                    imsave(default_path, img)
                    # add labels in napari gui
                    self._viewer.add_labels(masks, name=name)
                
                return {
                    "status": "success",
                    "mask_file_path": f"/tmp/{name}_mask_*",
                    "masks": masks,
                    "number_of_masks": len(masks)
                    
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to segment the image using cellpose"
            }
