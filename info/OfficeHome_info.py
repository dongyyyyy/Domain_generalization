domain_dic = {'Art' : 0, 'Clipart' : 1 , 'Product' : 2, 'RealWorld' :3}
class_dic_officeHome={'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 
        'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 
        'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 
        'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 
        'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 
        'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 
        'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}

        
def officeHome_dic():
       dataset_dic_officeHome = {'Art':{'Alarm_Clock': [], 'Backpack': [], 'Batteries': [], 'Bed': [], 'Bike': [], 'Bottle': [], 'Bucket': [], 'Calculator': [], 
              'Calendar': [], 'Candles': [], 'Chair': [], 'Clipboards': [], 'Computer': [], 'Couch': [], 'Curtains': [], 'Desk_Lamp': [], 'Drill': [], 
              'Eraser': [], 'Exit_Sign': [], 'Fan': [], 'File_Cabinet': [], 'Flipflops': [], 'Flowers': [], 'Folder': [], 'Fork': [], 'Glasses': [], 
              'Hammer': [], 'Helmet': [], 'Kettle': [], 'Keyboard': [], 'Knives': [], 'Lamp_Shade': [], 'Laptop': [], 'Marker': [], 'Monitor': [], 
              'Mop': [], 'Mouse': [], 'Mug': [], 'Notebook': [], 'Oven': [], 'Pan': [], 'Paper_Clip': [], 'Pen': [], 'Pencil': [], 'Postit_Notes': [], 
              'Printer': [], 'Push_Pin': [], 'Radio': [], 'Refrigerator': [], 'Ruler': [], 'Scissors': [], 'Screwdriver': [], 'Shelf': [], 'Sink': [], 
              'Sneakers': [], 'Soda': [], 'Speaker': [], 'Spoon': [], 'TV': [], 'Table': [], 'Telephone': [], 'ToothBrush': [], 'Toys': [], 'Trash_Can': [], 'Webcam': []}
                     ,'Clipart':{'Alarm_Clock': [], 'Backpack': [], 'Batteries': [], 'Bed': [], 'Bike': [], 'Bottle': [], 'Bucket': [], 'Calculator': [], 
              'Calendar': [], 'Candles': [], 'Chair': [], 'Clipboards': [], 'Computer': [], 'Couch': [], 'Curtains': [], 'Desk_Lamp': [], 'Drill': [], 
              'Eraser': [], 'Exit_Sign': [], 'Fan': [], 'File_Cabinet': [], 'Flipflops': [], 'Flowers': [], 'Folder': [], 'Fork': [], 'Glasses': [], 
              'Hammer': [], 'Helmet': [], 'Kettle': [], 'Keyboard': [], 'Knives': [], 'Lamp_Shade': [], 'Laptop': [], 'Marker': [], 'Monitor': [], 
              'Mop': [], 'Mouse': [], 'Mug': [], 'Notebook': [], 'Oven': [], 'Pan': [], 'Paper_Clip': [], 'Pen': [], 'Pencil': [], 'Postit_Notes': [], 
              'Printer': [], 'Push_Pin': [], 'Radio': [], 'Refrigerator': [], 'Ruler': [], 'Scissors': [], 'Screwdriver': [], 'Shelf': [], 'Sink': [], 
              'Sneakers': [], 'Soda': [], 'Speaker': [], 'Spoon': [], 'TV': [], 'Table': [], 'Telephone': [], 'ToothBrush': [], 'Toys': [], 'Trash_Can': [], 'Webcam': []},
                     'Product':{'Alarm_Clock': [], 'Backpack': [], 'Batteries': [], 'Bed': [], 'Bike': [], 'Bottle': [], 'Bucket': [], 'Calculator': [], 
              'Calendar': [], 'Candles': [], 'Chair': [], 'Clipboards': [], 'Computer': [], 'Couch': [], 'Curtains': [], 'Desk_Lamp': [], 'Drill': [], 
              'Eraser': [], 'Exit_Sign': [], 'Fan': [], 'File_Cabinet': [], 'Flipflops': [], 'Flowers': [], 'Folder': [], 'Fork': [], 'Glasses': [], 
              'Hammer': [], 'Helmet': [], 'Kettle': [], 'Keyboard': [], 'Knives': [], 'Lamp_Shade': [], 'Laptop': [], 'Marker': [], 'Monitor': [], 
              'Mop': [], 'Mouse': [], 'Mug': [], 'Notebook': [], 'Oven': [], 'Pan': [], 'Paper_Clip': [], 'Pen': [], 'Pencil': [], 'Postit_Notes': [], 
              'Printer': [], 'Push_Pin': [], 'Radio': [], 'Refrigerator': [], 'Ruler': [], 'Scissors': [], 'Screwdriver': [], 'Shelf': [], 'Sink': [], 
              'Sneakers': [], 'Soda': [], 'Speaker': [], 'Spoon': [], 'TV': [], 'Table': [], 'Telephone': [], 'ToothBrush': [], 'Toys': [], 'Trash_Can': [], 'Webcam': []},
                     'RealWorld':{'Alarm_Clock': [], 'Backpack': [], 'Batteries': [], 'Bed': [], 'Bike': [], 'Bottle': [], 'Bucket': [], 'Calculator': [], 
              'Calendar': [], 'Candles': [], 'Chair': [], 'Clipboards': [], 'Computer': [], 'Couch': [], 'Curtains': [], 'Desk_Lamp': [], 'Drill': [], 
              'Eraser': [], 'Exit_Sign': [], 'Fan': [], 'File_Cabinet': [], 'Flipflops': [], 'Flowers': [], 'Folder': [], 'Fork': [], 'Glasses': [], 
              'Hammer': [], 'Helmet': [], 'Kettle': [], 'Keyboard': [], 'Knives': [], 'Lamp_Shade': [], 'Laptop': [], 'Marker': [], 'Monitor': [], 
              'Mop': [], 'Mouse': [], 'Mug': [], 'Notebook': [], 'Oven': [], 'Pan': [], 'Paper_Clip': [], 'Pen': [], 'Pencil': [], 'Postit_Notes': [], 
              'Printer': [], 'Push_Pin': [], 'Radio': [], 'Refrigerator': [], 'Ruler': [], 'Scissors': [], 'Screwdriver': [], 'Shelf': [], 'Sink': [], 
              'Sneakers': [], 'Soda': [], 'Speaker': [], 'Spoon': [], 'TV': [], 'Table': [], 'Telephone': [], 'ToothBrush': [], 'Toys': [], 'Trash_Can': [], 'Webcam': []}}
       return dataset_dic_officeHome