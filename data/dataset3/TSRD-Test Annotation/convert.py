import pybboxes as pbx
import os
with open("TsignRecgTest1994Annotation.txt") as fp:
    for line in fp:
        file_name = line.split(";")[0].split(".")[0]
        coordinates = line.split(";")[1:-1]
        sign_coords = (int(coordinates[2]), int(coordinates[3]), int(coordinates[4]), int(coordinates[5]))
        yolo_coordinates = pbx.convert_bbox(sign_coords, from_type="voc", to_type="yolo", image_size=(int(coordinates[0]),int(coordinates[1])))
        yolo_coordinates_str = " ".join([str(value) for value in yolo_coordinates])
        result = f"0 {yolo_coordinates_str}"
        with open(os.path.join('./yolo_annotations', f"{file_name}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(result))
        print(result)