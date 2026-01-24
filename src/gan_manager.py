import os
import torch
import numpy as np
from utils.lightweight_gan_cust import Generator


class GANManager:
    # Aggiungiamo il parametro use_gpu=True come default
    def __init__(self, model_path, image_size=256, latent_dim=256, use_gpu=True):
        self.model_path = model_path
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # LOGICA DI SELEZIONE DEVICE
        # Usa CUDA solo se l'utente lo vuole E se è disponibile
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 GAN Manager: Modalità GPU (CUDA) attivata.")
        else:
            self.device = torch.device('cpu')
            if use_gpu and not torch.cuda.is_available():
                print("⚠️ GAN Manager: GPU richiesta ma non trovata. Fallback su CPU.")
            else:
                print("🐌 GAN Manager: Modalità CPU forzata.")

        # Carica il modello
        self.model = self._load_model()

    def _load_model(self):
        # ... (Il resto del codice rimane uguale, userà self.device automaticamente) ...
        # Copia pure il metodo _load_model e generate_image dal messaggio precedente
        # Assicurati solo che _load_model usi self.device come faceva prima
        print("Caricamento modello in corso...")
        
        model = Generator(
            image_size=self.image_size, 
            latent_dim=self.latent_dim
        )

        if not os.path.exists(self.model_path):
            print(f"⚠️ ERRORE: File modello non trovato in {self.model_path}")
            return model

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            gan_weights = checkpoint['GAN']

            gen_weights = {}
            found_ema = False
            
            for k, v in gan_weights.items():
                if k.startswith('GE.'):
                    gen_weights[k.replace('GE.', '')] = v
                    found_ema = True
            
            if not found_ema:
                print("Pesi EMA non trovati, uso pesi Base (G)...")
                for k, v in gan_weights.items():
                    if k.startswith('G.') and not k.startswith('GE.') and 'scaler' not in k:
                        gen_weights[k.replace('G.', '')] = v
            else:
                print("Pesi EMA (GE) trovati e caricati.")

            model.load_state_dict(gen_weights, strict=False)
            model.to(self.device)
            model.train() 
            
            print("✅ Modello caricato e pronto!")
            return model

        except Exception as e:
            print(f"❌ Errore critico nel caricamento: {e}")
            return model

    def generate_image(self, audio_volume) -> np.uint8:
        if self.model is None:
            return None

        with torch.no_grad():
            boosted_vol = audio_volume * 5.0 
            noise_amp = 0.5 + min(boosted_vol, 1.5) 

            noise = torch.randn(1, self.latent_dim).to(self.device) * noise_amp
            
            generated_tensor = self.model(noise)

            if torch.isnan(generated_tensor).any():
                generated_tensor = torch.nan_to_num(generated_tensor, nan=0.0)

            img_data = (generated_tensor.clamp(-1, 1) + 1) / 2
            img_data = (img_data * 255).byte()
            final_image = img_data[0].permute(1, 2, 0).cpu().numpy()
            
            return final_image