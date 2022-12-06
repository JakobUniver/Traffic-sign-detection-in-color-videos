import json
import pybboxes as pbx
import os

input_path = "./annotations/"
with os.scandir(input_path) as files:
    for file in files:
        with open(f"./annotations/{file.name}", 'r') as fcc_file:
            print(file.name.split(".")[0])
            data = json.load(fcc_file)
            output_filename = file.name.split(".")[0]
            with open(os.path.join('./yolo_annotations/', f"{output_filename}.txt"), "w", encoding="utf-8") as output_f:
                for object in data["objects"]:
                    sign_coords = (object["bbox"]["xmin"], object["bbox"]["ymin"], object["bbox"]["xmax"], object["bbox"]["ymax"])
                    yolo_coordinates = pbx.convert_bbox(sign_coords, from_type="voc", to_type="yolo", image_size=(data["width"],data["height"]))
                    yolo_coordinates_str = " ".join([str(value) for value in yolo_coordinates])
                    result = f"0 {yolo_coordinates_str}\n"
                    print(result)
                    output_f.write(result)
#     yolo_coordinates = pbx.convert_bbox(sign_coords, from_type="voc", to_type="yolo", image_size=(data["width"],data["height"]))
#     yolo_coordinates_str = " ".join([str(value) for value in yolo_coordinates])
#     result = f"0 {yolo_coordinates_str}"
#     with open(os.path.join('./yolo_annotations', f"Bh36Ed4HBJatMpSNnFTgTw.txt"), "w", encoding="utf-8") as f:
#                 f.write("\n".join(result))
#     print(result)