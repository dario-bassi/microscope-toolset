import ast
import re

from local.execute import Execute


class MicroscopeStatus:
    def __init__(self, executor: Execute):
        self.executor = executor

    def get_current_status(self) -> dict:
        code = "print(mmc.getSystemState().dict())"
        result_status = self.executor._run_code_live(code)
        # with the virtual microscope, maybe we get as key a python object
        # because of this we clean it
        cleaned_string = re.sub(r"<[^:]+:\s+'([^']+)'\s*>", r"'\1'", result_status)
        try:
            current_status = ast.literal_eval(cleaned_string)
        except (ValueError, SyntaxError):
            return {"error": f"Failed to parse status: {cleaned_string}"}
        return current_status

    def get_properties(self) -> dict:
        code = "tmp_list=[(x, mmc.getDeviceSchema(x)) for x in mmc.getLoadedDevices()]\nall_properties = dict(tmp_list)\nprint(all_properties)"
        result_properties = self.executor._run_code_live(code)
        # with the virtual microscope, maybe we get as key a python object
        # because of this we clean it
        cleaned_string = re.sub(r"<[^:]+:\s+'([^']+)'\s*>", r"'\1'", result_properties)
        try:
            json_object = ast.literal_eval(cleaned_string)
        except (ValueError, SyntaxError):
            return {"error": f"Failed to parse properties: {cleaned_string}"}
        return json_object

    def get_available_configs(self):
        code = "complete_dict={}\nfor x in mmc.getAvailableConfigGroups():\n\ttmp_dict={}\n\tfor y in mmc.getAvailableConfigs(x):\n\t\tconfiguration = mmc.getConfigData(x,y).dict()\n\t\ttmp_dict[y] = configuration\n\tcomplete_dict[x]=tmp_dict\nprint(complete_dict)"

        available_config = self.executor._run_code_live(code)
        # with the virtual microscope, maybe we get as key a python object
        # because of this we clean it
        cleaned_string = re.sub(r"<[^:]+:\s+'([^']+)'\s*>", r"'\1'", available_config)
        try:
            config_json = ast.literal_eval(cleaned_string)
        except (ValueError, SyntaxError):
            return {"error": f"Failed to parse configurations: {cleaned_string}"}
        return config_json
