This is code deploys the generator (african fabric dataset) to the localhost. Both the GET and POST requests are successful on Postman. On the browser, the get request is successful but post request fails (405 method not allowed error). I looked at https://airbrake.io/blog/http-errors/405-method-not-allowed
but too many possibilities - I do not know how to debug.

To run this code :
conda env create --file model\conda.yaml
conda activate mlflow-env-fabric_vision

Once inside the active conda environment,
python app.py

Open postman. 
Send GET request by clicking on "send" (localhost:5000/). The following should appear :
{
  "description": "The generator part of trained GAN. state_dict()",
  "input": "The predict method does not need an input. It generates random numbers which are the input to the generator. Every invocation should generate different set of 100 random numbers.",
  "version": "1.0"
}

Sent a POST request by clicking on "send" (localhost:5000/predict). A 8x8 grid of african fabric designs should appear. Clicking the
"send" button repeatedly should display a different set of 64 images. Make sure that the input is set to "none" in "Body" while sending the request.

A screenshot of the postman is shown in african_fabric_deploy.png
