from torchsummary import summary
import os
from torchviz import make_dot
import torch
import torch.nn as nn



#Show architecture of given model
def Show_encoder_summary(encoder, input_tensor, save_png=False, folder="Model_graphs", name="Model"):
    print("Preparing model...")
    
    
    # rapped model with preset parameter
    
    encoder.to("cpu")
    encoder.eval()
    
    shape = input_tensor.shape
    
    summary(encoder, shape, device="cpu")  #Print summary


    if save_png:
        #Ensure folder exists
        os.makedirs(folder, exist_ok=True)
        
        #Output tensor of network in given state
        output_tensor, _ = encoder(input_tensor.unsqueeze(0))

        #Generate computation graph
        graph = make_dot(output_tensor, params=dict(encoder.named_parameters()))

        #Save graph png
        graph.format = "png"
        save_path = os.path.join(folder, name)  # Ensure correct naming
        graph.render(save_path)
        print(f"Model architecture saved as '{save_path}.png'")
        
        del output_tensor,graph
        
    del input_tensor
    del encoder
    torch.cuda.empty_cache()

    
    
import os
import torch
from torchsummary import summary
from torchviz import make_dot

def Show_decoder_summary(decoder, encoder, input_tensor, class_vector_size=1,
                         device="cpu", save_png=False, folder="Model_graphs", name="Decoder_Model"):
    """
    Shows a decoder summary using torchsummary and optionally saves computation graph as PNG.
    
    Args:
        decoder      : PyTorch decoder model
        encoder      : PyTorch encoder model (used to generate dummy bottleneck + skips)
        input_tensor : single tensor to feed to encoder for generating shapes, e.g., torch.zeros(3,256,256)
        class_vector_size: size of class vector (default 1)
        device       : device to run on, e.g., 'cpu' or 'cuda'
        save_png     : whether to save computation graph as PNG
        folder       : folder to save PNG
        name         : file name for PNG (without extension)
    """
    
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder, encoder, input_tensor, class_vector_size):
            super().__init__()
            self.decoder = decoder
            self.class_vector_size = class_vector_size
            
            # Generate shapes using encoder
            encoder.eval()
            with torch.no_grad():
                dummy_input = input_tensor.unsqueeze(0).to(device)  # add batch dim
                bottleneck, skips = encoder(dummy_input)
                self.bottleneck_shape = tuple(bottleneck.shape[1:])
                self.skip_shapes = [tuple(s.shape[1:]) for s in skips]
            
        def forward(self, bottleneck):
            batch_size = bottleneck.shape[0]
            skips = [torch.zeros(batch_size, *s, device=bottleneck.device) for s in self.skip_shapes]
            class_vector = torch.zeros(batch_size, self.class_vector_size, device=bottleneck.device)
            return self.decoder(bottleneck, *skips, class_vector)
    
    # Wrap decoder
    wrapper = DecoderWrapper(decoder, encoder, input_tensor, class_vector_size)
    wrapper.to(device)
    wrapper.eval()
    
    print("==== Decoder Summary ====")
    summary(wrapper, input_size=(wrapper.bottleneck_shape), device=device)
    
    # Save PNG if requested
    if save_png:
        os.makedirs(folder, exist_ok=True)
        
        # Forward a dummy tensor to get graph
        dummy_input = torch.zeros(1, *wrapper.bottleneck_shape).to(device)
        output_tensor = wrapper(dummy_input)
        
        graph = make_dot(output_tensor, params=dict(wrapper.named_parameters()))
        save_path = os.path.join(folder, name)
        graph.format = "png"
        graph.render(save_path)
        print(f"Decoder computation graph saved as '{save_path}.png'")
        
        # Cleanup
        del output_tensor, graph
        torch.cuda.empty_cache()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    