from ultralytics import YOLO

model = YOLO("yolo11n.pt")
name_model = 'yolo11_base_dataset'

results = model.train(data="config_yolo.yaml",
                      device="mps",
                        epochs=100,
                        #patience=20,
                        imgsz=160 ,
                        workers=8,
                        batch=0.9,
                        save_period=50,
                        project='models',
                        name=name_model,
                        # rect=True,
                        plots=True,
                        )
                        
print(f"Treinamento do datasetfinalizado e salvo em models/{name_model}")