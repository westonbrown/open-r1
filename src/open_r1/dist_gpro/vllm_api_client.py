import requests
import time

class VllmApiClient:
    """
    A client class for interacting with the vLLM FastAPI server.
    """

    def __init__(self, server_ip:str, server_port:str):
        """
        Initializes the VllmApiClient.

        Args:
            server_ip: IP of the FastAPI server.
            server_port: port of the FastAPI server.
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_url = f"http://{server_ip}:{server_port}"
        self.vllm_url = None

    def launch_model(self, model_name: str, vllm_port:int):
        """
        Sends a request to the FastAPI server to launch a vLLM model.

        Args:
            model_name: The name of the model to launch (e.g., "facebook/opt-125m").
        """
        endpoint = f"{self.server_url}/launch"
        params = {"model": model_name, "vllm_port": vllm_port}
        try:
            response = requests.post(endpoint, params=params)
            response.raise_for_status()
            print(f"Successfully launched model: {model_name}")
            print(response.json())
            self.vllm_url =  f"http://{self.server_ip}:{vllm_port}"
        except requests.exceptions.RequestException as e:
            print(f"Error launching model: {e}")

    def wait_for_ready(self):
        """
        Wait for model to be ready
        """
        start_time = time.time()
        endpoint = f"{self.server_url}/wait_for_ready"
        try:
            response = requests.post(endpoint)
            response.raise_for_status()
            print(f"Model is ready in {time.time() - start_time} seconds")
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error launching model: {e}")

    def stop_server(self):
        """
        Sends a request to the FastAPI server to stop the vLLM process.
        """
        endpoint = f"{self.server_url}/stop"
        try:
            response = requests.post(endpoint)
            response.raise_for_status()
            print("Successfully stopped vLLM process")
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error stopping vLLM process: {e}")

    def stop_and_launch_model(self, model_name: str, vllm_port:int):
        """
        Sends a request to the FastAPI server to stop the current vLLM process and launch a new one.

        Args:
            model_name: The name of the new model to launch.
        """
        endpoint = f"{self.server_url}/stop_and_launch"
        params = {"model": model_name, "vllm_port": vllm_port}

        try:
            response = requests.post(endpoint, params=params)
            response.raise_for_status()
            print(f"Successfully stopped current process and launched model: {model_name}")
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error stopping and launching model: {e}")

    def check_health(self):
        """
        Sends a request to the FastAPI server to check the health of the vLLM instance.
        """
        endpoint = f"{self.server_url}/health"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            print("vLLM instance is healthy")
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"vLLM instance is not healthy: {e}")

    def get_status(self):
        """
        Sends a request to the FastAPI server to get the status of the vLLM handler.
        """
        endpoint = f"{self.server_url}/status"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            print("vLLM handler status:")
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error getting vLLM handler status: {e}")

# Example usage:
if __name__ == "__main__":
    client = VllmApiClient(server_url="http://0.0.0.0:8000")

    # 1. Launch the first model
    client.launch_model("facebook/opt-125m")
    client.wait_for_ready()
    # 2. Check health
    client.check_health()

    # 3. Get status
    client.get_status()

    # 4. Stop and launch a new model
    client.stop_and_launch_model("facebook/opt-350m")
    client.wait_for_ready()

    # 5. Check health again
    client.check_health()

    # 6. Get status again
    client.get_status()

    # 7. Stop the server
    client.stop_server()