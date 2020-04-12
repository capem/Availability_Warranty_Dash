# Availability-Warranty-Dash

## Getting Started
-  Download this project and extract in desired location.

- Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python version 3.7 from Anaconda website.

- Open anaconda prompt in windows start menu and run these commands.

    ```bash
    conda create --name dash-env -y python
    conda activate dash-env
    ```

- In the same anaconda prompt navigate to the project folder location

    - Example : if project is extracted in downloads:
        ```bash
        cd ./Downloads/"Availability Warranty Dash"
        pip install -r requirements.txt
        ```

- If you are using a 64-bit version of Python but you have the 32-bit version of the Access Database Engine installed. You either need to :
    - run a 32-bit version of Python, or
    - remove the 32-bit version of the Access Database Engine and install the  64-bit version (available [here](https://www.microsoft.com/en-US/download/details.aspx?id=13255)).


- Launch WebApp
    ```bash
    python index.py
    ```

