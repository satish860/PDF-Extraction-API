# PDF Extraction API

PDF Extraction API is a simple API built on top of the Marker PDF library. It allows you to parse and convert PDF files into Markdown format, extract images, and retrieve the parsed content through local and web endpoints.

## Installation

1. Clone the repository: `git clone [REPOSITORY_URL]`
2. Install the required dependencies: `pip install modal marker-pdf hf_transfer opencv-python-headless pillow`

## Usage

### Local Endpoint

To run the PDF parser and converter locally, use the following command:

`modal run app.py`

This will execute the `main` function defined in `app.py`, which downloads a PDF from a specified URL, processes it using Marker PDF, and saves the parsed content and images locally.

### Web Endpoint

To run the PDF parser and converter as a web endpoint, use the following command:

`modal serve app.py`

This will start a web server and expose the `/convert` endpoint. You can send a POST request to this endpoint with the PDF chunk as a base64-encoded string in the request body. The endpoint will process the PDF using Marker PDF and return the parsed Markdown content, base64-encoded images, and metadata as a JSON response.

## Configuration

Before running the application, make sure to set the following environment variables:


`export HF_HUB_ENABLE_HF_TRANSFER=1`


`export TRANSFORMERS_CACHE=/data/transformers_cache`


`export HF_HOME=/data/hf_home`


These environment variables are required for the proper functioning of the Marker PDF library.

## License

This project is licensed under the [MIT License](LICENSE).
