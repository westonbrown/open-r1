
from fastapi import FastAPI, HTTPException

import subprocess
import psutil
import time
import os
import uvicorn
from typing import List, Optional, Dict
import requests

class VllmHandler:
    """
    A class to handle the starting, stopping, and management of a vLLM process.
    """

    def __init__(self):
        self.process = None
        self.current_model: str = None  # Keep track of the currently loaded model
        self.current_port = None
        self.log_file_handle = open("vllm.log", "a")
        

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def launch(self, model: str, port:int):
        if self.is_running():
            if self.current_model == model and self.current_port== port:
                print(f"vLLM process is already running with model: {model}")
                return
            else:
                self.stop()  # Stop the current process if a different model is requested

        launch_command = [
            "vllm",
            "serve",
            model,
            "--port",
            str(port)
        ]
        print(f"Launching vLLM process with model: {model}...{port}")
        try:
            self.process = subprocess.Popen(
                launch_command,
                stdout=self.log_file_handle,  # Redirect stdout to the log file
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            )
            self.current_model = model
            self.current_port = port
            print(f"vLLM process launched with PID: {self.process.pid}")
        except Exception as e:
            print(f"Error launching vLLM process: {e}")
            self.process = None
            self.current_model = None
            self.current_port = None
            
    def _is_healthy(self, timeout: int = 5) -> bool:
        """
        Checks if the vLLM instance is healthy by making a request to the health endpoint.

        Args:
            timeout: The timeout in seconds for the health check request.

        Returns:
            True if the instance is healthy, False otherwise.
        """
        try:
            print("querying ", f"http://0.0.0.0:{self.current_port}/health")
            response = requests.get(f"http://0.0.0.0:{self.current_port}/health", timeout=timeout)
            print(response)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


    def wait_for_ready(self, timeout: float = 120.0, retry_interval: float = 2.0) -> bool:
        """
        Waits for the vLLM instance to become ready within a specified timeout.

        Args:
            timeout: The maximum time in seconds to wait for the instance to become ready.
            retry_interval: The time in seconds to wait between retries.

        Returns:
            True if the instance becomes ready within the timeout, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_healthy():
                print("vLLM instance is ready.")
                return True
            else:
                print(f"vLLM instance is not ready yet. Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
        print(f"vLLM instance did not become ready within {timeout} seconds.")
        return False
    
    def stop_and_launch(self, new_model:str, new_port:int):
        self.stop()
        self.launch(new_model, new_port)

    def stop(self, timeout: int = 10):
        """
        Stops the vLLM process gracefully, then forcefully if needed.

        Args:
            timeout: The number of seconds to wait for graceful termination 
                     before force killing.
        """
        if not self.is_running():
            print("vLLM process is not running.")
            return

        print(f"Stopping vLLM process (PID: {self.process.pid})...")
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
            print("vLLM process stopped gracefully.")
        except subprocess.TimeoutExpired:
            print(f"Graceful termination timed out after {timeout} seconds. Force killing...")
            self.process.kill()
            self.process.wait()
            print("vLLM process force killed.")
        except Exception as e:
            print(f"Error stopping vLLM process: {e}")
        finally:
            self.process = None
            self.current_model = None

# Create a FastAPI app
app = FastAPI()

# Create an instance of VllmHandler
handler = VllmHandler()

# --- FastAPI Endpoints ---

@app.post("/launch")
def launch_model(model: str, vllm_port:int):
    """
    Launches the vLLM process with the specified model.
    """
    try:
        handler.launch(model, vllm_port)
        return {"message": f"vLLM process launched with model: {model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_model():
    """
    Stops the currently running vLLM process.
    """
    try:
        handler.stop()
        return {"message": "vLLM process stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_and_launch")
def stop_and_launch_model(model: str, vllm_port:int):
    """
    Stops the currently running vLLM process and launches a new one with the specified model.
    """
    try:
        handler.stop_and_launch(model, vllm_port)
        return {"message": f"vLLM process stopped and new model launched: {model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/wait_for_ready")
def wait_for_ready():
    """
    Stops the currently running vLLM process and launches a new one with the specified model.
    """
    try:
        if handler.wait_for_ready():
            return {"message": f"vLLM process is ready"}
        else:
            raise HTTPException(status_code=500, detail="vLLM instance did not become ready")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def check_health():
    """
    Checks the health of the vLLM instance.
    """
    if handler._is_healthy():
        return {"message": "vLLM instance is healthy"}
    else:
        raise HTTPException(status_code=503, detail="vLLM instance is not healthy")

@app.get("/status")
def get_status():
    """
    Gets the current status of the vLLM handler.
    """
    return {
        "running": handler.is_running(),
        "model": handler.current_model,
        "port": handler.current_port,
    }

# --- Run the FastAPI app with uvicorn ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)