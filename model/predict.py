# Import necessary libraries 
from model import Model
from IPython.display import display


# Function for predicting data 
def predict_data(data=None):
    # Alias the data
    pred_data = data

    final_pred_result = "Not available"
    loaded_model = None
    loaded_label = None

    try:
        # Load pretrained model file 
        loaded_model = Model.rf_model
        loaded_label = Model.encoder_target
        print(f"File named {loaded_model} and {loaded_label} successfully loaded.")

        # Predict the final status based on given data 
        if loaded_model is not None and pred_data is not None:
            try:
                pred_result = loaded_model.predict(pred_data) 
                final_pred_result = loaded_label.inverse_transform(pred_result)[0]
                display(final_pred_result)
            except Exception as e:
                print(f"An error occured while running prediction: {e}")
                final_pred_result = "Prediction failed" 

            # Return the value of 'Status' ('final_pred_result') whatever the results 
            return final_pred_result

        else:
            # This will be executed if the model fails to predict 
            print("\nFail to load model or unsuitable data found. No prediction given at this time...")

    except FileNotFoundError:
        print(f"File named {loaded_model} or {loaded_label} not found.")

    except Exception as e:
        print(f"Fail to load model: {e}")