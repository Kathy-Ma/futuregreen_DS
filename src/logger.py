from datetime import datetime
import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, logging_folder_name):
        self.logging_source_path = "logs/"
        os.makedirs(self.logging_source_path, exist_ok=True)

        now = datetime.now()
        logging_folder_name_timestamped = f"{logging_folder_name}_{now.strftime('%m-%d-%Y_%H-%M-%S')}_logs"
        self.logging_folder_path = os.path.join(self.logging_source_path, logging_folder_name_timestamped)

        os.makedirs(self.logging_folder_path)
        print(self.logging_folder_path)

        self.logging_file_path = os.path.join(self.logging_folder_path, "log.txt")
        open(self.logging_file_path, "a").close()  # create file if it doesn't exist
    


    def log_message(self, message):
        """
        Logs a message to the logging folder
        Args:
            message (str): the message to log
        Returns:
            None
        """
        with open(self.logging_file_path, "a") as f:
            f.write(message + "\n")



    def save_figure(self, filename):
        """
        Saves the current matplotlib figure to the logging folder.
        Args:
            filename (str): the filename
        Returns:
            None
        """
        path = os.path.join(self.logging_folder_path, filename)
        plt.savefig(path)
        plt.close()



    def save_image(self, image, image_name):
        """
        Saves an image (numpy array) to the logging folder.
        Args:
            image (numpy.ndarray): the image to save (BGR or RGB)
            image_name (str): the filename (e.g. "sample.jpg")
        Returns:
            None
        """
        import cv2
        path = os.path.join(self.logging_folder_path, image_name)
        cv2.imwrite(path, image)