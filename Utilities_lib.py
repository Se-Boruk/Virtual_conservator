from torchsummary import summary
import os
from torchviz import make_dot
import torch

#Show architecture of given model
def Show_architecture(model, input_tensor, save_png=False, folder="Model_graphs", name="Model"):
    print("Preparing model...")
    # rapped model with preset parameter
    
    model.to("cpu")
    model.eval()
    input_tensor = input_tensor.to("cpu")  #Ensure correct tensor assignment
    
    input_shape = input_tensor.shape

    summary(model, input_shape, device="cpu")  #Print summary


    if save_png:
        #Ensure folder exists
        os.makedirs(folder, exist_ok=True)
        
        #Output tensor of network in given state
        output_tensor = model(input_tensor.unsqueeze(0))

        #Generate computation graph
        graph = make_dot(output_tensor, params=dict(model.named_parameters()))

        #Save graph png
        graph.format = "png"
        save_path = os.path.join(folder, name)  # Ensure correct naming
        graph.render(save_path)
        print(f"Model architecture saved as '{save_path}.png'")
        
        del output_tensor,graph
        
    del input_tensor
    del model
    torch.cuda.empty_cache()