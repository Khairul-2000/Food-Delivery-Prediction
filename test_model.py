import torch
import torch.nn as nn
# from model import distances_mean, distances_std, times_mean, times_std



checkpoint = torch.load("model_bundle.pth")


distance_to_predict = 5.1


# Recreate the same model structure
loaded_model = nn.Sequential(nn.Linear(1, 3),
                             nn.ReLU(),
                             nn.Linear(3, 1))

# Load weights
loaded_model.load_state_dict(checkpoint["model_state"])

# Set to evaluation mode
loaded_model.eval()



test_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
# prediction = loaded_model(test_distance)




# Use the torch.no_grad() context manager for efficient prediction
with torch.no_grad():
    # Normalize the input distance
    distance_tensor = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    new_distance_norm = (distance_tensor - checkpoint["dist_mean"]) / checkpoint["dist_std"]
    
    # Get the normalized prediction from the model
    predicted_time_norm = loaded_model(new_distance_norm)
    
    # De-normalize the output to get the actual time in minutes
    predicted_time_actual = (predicted_time_norm * checkpoint["time_std"]) + checkpoint["time_mean"]
    
    # --- Decision Making Logic ---
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time_actual.item():.1f} minutes")
    
    # First, check if the delivery is possible within the 45-minute timeframe
    if predicted_time_actual.item() > 45:
        print("\nDecision: Do NOT promise the delivery in under 45 minutes.")
    else:
        # If it is possible, then determine the vehicle based on the distance
        if distance_to_predict <= 3:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (<= 3 miles), use a bike.")
        else:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distance_to_predict} miles (> 3 miles), use a car.")