import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer

from audio_manager import AudioManager
from gui import GUI
from gan_manager import GANManager
from utils.custom_enum import FPS, SampleWindowSize

### CUSTOM VARIABLES ###

# Set Visualizer FPS
FRAMERATE = FPS.FPS_30
# Set sample window size to retrieve real time from the audio
SAMPLE_WINDOW_SIZE = SampleWindowSize.WS_1024

MODEL_PATH = './resources/models/light_gan_test/model_3.pt'
USE_GPU = False

########################

class VisualizerApp:
    def __init__(self):
        # Inizializza audio e gui managers
        self.audio_system = AudioManager()
        self.window = GUI(self.audio_system, img_size=256)

        self.gan_manager = GANManager(
            model_path=MODEL_PATH, 
            image_size=256, 
            latent_dim=256,
            use_gpu=USE_GPU
        )

        if self.gan_manager is None:
            return
        
        # Configura il timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(FRAMERATE)

        # Avvia la GUI
        self.window.show()

    def update_loop(self):
        # Se il player non è in Play (è in Pausa o Fermo), non generare nulla
        if self.audio_system.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return

        # Recupero un chunk audio di una finestra temporale
        chunk = self.audio_system.get_current_chunk(window_size=SAMPLE_WINDOW_SIZE)

        # test con calcolo volume real time 
        volume = np.linalg.norm(chunk)

        final_image = self.gan_manager.generate_image(audio_volume=volume)
        if final_image is not None:
            self.window.set_image(final_image)

        # TODO: Invia i dati a TouchDesigner



def main():
    app = QApplication(sys.argv)
    controller = VisualizerApp()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()