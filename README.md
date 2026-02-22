CropShield - Crop Yield Prediction and Risk Alert System

A Python-based system that predicts crop yield based on weather, soil, and crop data. It also provides risk alerts to farmers to improve crop management and sustainability.

Features
- Predict crop yield using machine learning.
- Fetch real-time weather data using latitude and longitude.
- Detect crop type and provide crop-specific alerts.
- User-friendly GUI for easy interaction.

Technologies Used
- Python 3
- Libraries: pandas, scikit-learn, tkinter, requests
- API: OpenWeatherMap

Installation
1. Open Command Prompt (CMD)  
2. Navigate to project folder:
   ```cmd
   cd C:\CropShield
   dir
   pip install -r requirements.txt
   python app.py
   
Usage
Enter the crop type, location (latitude & longitude), and soil details.

Click "Predict" (GUI) or press Enter (console) to see the estimated yield and risk alerts.

SAMPLE OUTPUT
Enter crop type: Paddy
Enter latitude: 12.9716
Enter longitude: 77.5946
Enter soil moisture (%): 35

Predicted Paddy yield: 2.5 tons/hectare
Risk Alert: Low soil moisture, irrigation recommended

<img width="1920" height="1080" alt="Screenshot 2026-02-22 175243" src="https://github.com/user-attachments/assets/880887be-69b0-406a-9783-521027895ef5" />

