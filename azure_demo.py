print('Importing modules and starting trainer...')



#Import various tools
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

#create a project and resources online, then enter the details here
ENDPOINT = "https://educ463b.cognitiveservices.azure.com/"
training_key = "7483a7108873428ba1e7197a24fc0f71"
prediction_key = "62fcf010ccba49d58aebee8a37def6b9"
prediction_resource_id = "/subscriptions/95f478e7-5b73-439d-8536-16dc258609f6/resourceGroups/demo/providers/Microsoft.CognitiveServices/accounts/educ463b-Prediction"
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

publish_iteration_name = "classifyModel"

# Create a new project
print ("Creating project...")
project_name = uuid.uuid4()
project = trainer.create_project(project_name)


# Make two tags in the new project
# seamus_tag = trainer.create_tag(project.id, "seamus")
# finley_tag = trainer.create_tag(project.id, "finley")

base_image_location = '/Users/nickhaber/Downloads/teachable_machine_demo'

print("Adding images...")


#Need to create a list of all ImageFileCreateEntry objects
#with images tagged according to the label
#the control flow here reflects my directory structure
images = {}
filenames = {}
labels = ['seamus', 'finley']
modes = ['train', 'test']

for label_ in labels:
    images[label_] = {}
    filenames[label_] = {}
    tag_ = trainer.create_tag(project.id, label_)
    for mode_ in modes:
        images[label_][mode_] = []
        dir_ = os.path.join(base_image_location, label_, mode_)
        names_ = os.listdir(dir_)
        names_ = [os.path.join(dir_, nm_) for nm_ in names_]
        names_ = [nm_ for nm_ in names_ if os.path.isfile(nm_)]
        filenames[label_][mode_] = names_
        for nm_ in names_:
        	with open(nm_, 'rb') as img_:
        		images[label_][mode_].append(ImageFileCreateEntry(name = nm_, 
        			contents = img_.read(),
        			tag_ids = [tag_.id]))

#upload the training data to azure
upload_result = trainer.create_images_from_files(
    project.id, ImageFileCreateBatch(images=images['seamus']['train'] + images['finley']['train']))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    exit(-1)


#train! It's as simple as that. Might take a few minutes...
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)


# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")


# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

#testing
#the first five of these should be finley, the second five should be seamus
print('Should predict finley')
for filename_ in filenames['finley']['test']:
    with open(filename_, 'rb') as img_:
        results = predictor.classify_image(project.id, publish_iteration_name, img_.read())
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))

print('Should predict seamus')
for filename_ in filenames['seamus']['test']:
    with open(filename_, 'rb') as img_:
        results = predictor.classify_image(project.id, publish_iteration_name, img_.read())
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))
