class InvalidConfigValue(RuntimeError):
    def __init__(self, configuration_name, value: str):
        self.configuration_nam = configuration_name
        self.value = value
        super().__init__(f"Invalid config value for {configuration_name}: {value}")


class AdapterNotSupported(RuntimeError):
    def __init__(self, adapter_type: str, framework_name: str):
        self.adapter_type = adapter_type
        self.framework_name = framework_name
        super().__init__(f"Not supported adapter type {adapter_type} for framework {framework_name}")
