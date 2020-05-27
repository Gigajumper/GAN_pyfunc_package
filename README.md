# GAN_pyfunc_package
Trying out packaging of generator using the example of african fabric design.

fabric_gan_package.ipynb : contains the training code.
model.py : contains the generator code

I moved all the generator related stuff to model.py.
However I get the error "name 'F' is not defined" while executing netG(Fake).

I wrote a basic 
ModelWrapper(mlflow.pyfunc.PythonModel):

This needs further modification. It is adapted from teh IrisNet example.
