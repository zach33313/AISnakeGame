INSTRUCTIONS ON HOW TO SET UP THE PROJECT

Overview: 
  This repo has files of 3 types, Frontend(any .tsx or .css or image files), Backend(app.py), and Training files(any .py file that is not app.py). 
    1. Frontend files are made using react and typescript. The structure of the fe is as follow App.tsx is a route to either AppGame(the game) or ModelView(a tool I made to see the training progess). ALL FILES GO in the frontend NEED to be in /src/ in your react app. 
    2. Backend is just app.py, nothing major for the set up of this just follow instructions below
    3. The training files are all different AI algorithms with differing reward functions, if you want to train a model using a reward function PLEASE USE updated_train.py(this is an API built into the training logic, you need this for modelview)

    

Backend Set Up: 
    1. Make a folder called snake-be(or anything you want)
    2. cd into ur new folder and create a file called app.py(and copy app.py into it) 
    3. run these commands in order
        a. python3 -m venv path/to/venv
        b. source path/to/venv/bin/activate      (this activates your virtual enviornment) 
        c. pip3 install flask flask-cors flask-socketio eventlet torch numpy      (installing packages)
    4. Try running python3 app.py, if successful you should see a message like this 
        venv) zachhixson@Zachs-MacBook-Pro snake-back-end % python3 app.py
        (69182) wsgi starting up on http://123.0.0.1:5000 
    5. Copy the URL(http://123.0.0.1:5000) we need it for later, also your url will possible look different, that is ok! 
    6. Last thing, if ur getting an error about not being able to find a weird file path text me, its a model but I can't put it on github cuz its too big, I will share it w u via google drive and u can download it and put it in ur backend folder and it will work! Sorry about that.

Frontend Set Up: 
  1. outside of your backend(cd ..) run this npx create-react-app snake-front-end --template typescript
  2. This makes a react app called snake-front-end, this doesn't need to be the name btw you can call it anything
  3. We are now going to download the libaries we need( npm install socket.io-client axios react-chartjs-2 react-router-dom)
  5. First copy the App.tsx from github and put it in your App.tsx in your react project(THERE ARE GOING TO BE SO MANY ERRORS BUT ITS CHILL ONG) 
  5. Ok lets put our files in now, add and copy EVERY file that ends in .tsx or .css(except App.tsx, we just did that one) and put it in the /src/ folder in your react app
  6. Now download the image files from github and put them in your /src/ folder(DO NOT CHANGE THE NAMES OF THE IMAGES AND DO NOT PUT THEM IN A FOLDER THEIR FILE PATHS MUST BE /src/{image_name}.png)
  7. Ok we almost done, take the url you copied in Backend step 5 and copy and paste it into socket.tsx(i think I left a comment saying where to put it).
  8. alright bet we done I think. run da prog wit npm start
  9. you land on the AppGame page but u can switch to the model view by adding /mode_view to the url!


Train.py Set Up(optional):
  1. python3 updated_train.py
  2. get the url like in BE step 5, and put it into ModelView.tsx in the server1 variable(cmd f it twin)




Alright u done
    










