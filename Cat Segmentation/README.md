# Cat-Segmentation

Cat-Segmentation is a deep learning project designed to segment images of cats. The project uses a deep learning model to identify and segment specific regions in cat images.

Training: The model is trained using images of cats and their corresponding segmentation masks.
Testing: The trained model is used to segment new images of cats.

## How to Run

1. **Install Python Dependencies**:
   Install the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt

2. **Training the Model**:
   Use the main.py script to train the model. 
   ```bash
   python main.py
   ```
   The training process:
   Uses images from Data/input/ as input.
   Uses segmentation masks from Data/mask/.
   Saves the trained model as model.pth in the results/ folder.

3. **Training the Model**:
   Use the test.py script to test the trained model.
   ```bash
   python test.py
   ```
   The testing process:
   Loads the trained model from results/model.pth.
   Performs segmentation on images in the Data/Test_Data/ folder.
   Saves the segmented images to the Data/Result_Data/ folder.
