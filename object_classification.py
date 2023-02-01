# from Camera.results_camera.inference import camera_inference
# from Gripper.object_classification.Classifiers.tree_classifier import tree_class
from Gripper.object_classification.Classifiers.KNN import KNN
# from Gripper.object_classification.Classifiers.SVM import SVM

# SELECT THE ML ALGORITHM:
algorithm = KNN
metric = 'minkowski'
from Camera.results_camera.inference import camera_inference

pred_classes, orig_image = camera_inference()

print("The predicted objects are: ", pred_classes)


print("Write --robot-- to tell the terminal that the robot has reached the positioned to test tactile: ", )
x = input()
# print(f"{x}\n")

if x == "robot":
    if metric!="":
        result = algorithm(pred_classes[0], metric = metric)
    else:
        result = algorithm(pred_classes[0])
    
    print("The final identification is:\n", result)
else:
    print("ERROR, UNKNOWN DECISION")



