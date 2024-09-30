import speech_recognition as sr

r = sr.Recognizer()

def activate():
    with sr.Microphone() as source2:
            # Adjust the recognizer sensitivity to ambient noise
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            # Listen for the user's input 
            print("Listening...")
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print(f"You said: {MyText}")
    return MyText
