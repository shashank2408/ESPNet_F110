import json 
import cv2
import numpy as np
# f = open("layers.json")
# o = json.load(f)

# for item in o["Layers"]:
#     print (item["layer_name"])
#     print (item["objects"])


file = "/home/mlab-train/Desktop/deeprl/LaneDetection/EdgeNets/results_city_test/results/image_right000001.png"

img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
# np.savetxt("foo.csv", img, delimiter=",")
p = np.array(np.where(img == 21))
# p.append(list(np.where(img==11)))
# np.savetxt("foo_ind.csv", p, delimiter=",")
print(p)
print(np.hstack((p[0],p[1])))

print(img.shape)