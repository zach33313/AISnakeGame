For the Front-end: 

Make a react app(IN TYPESCRIPT called anything(npx create-react-app snake-front-end --template typescript). 
Take the App.tsx I have in github and replace the og App.tsx. Add restart.tsx and SnakeGame.tsx into the src folder in ur react app. 
As long as you have React and Typescript installed its chill 

***YOU WILL NEED TO REPLACE THE ip_and_port VARIABLE WITH WHERE UR BACKEND IS RUNNING*** but thats lwk the only change u need to make. 

to run the frontend go to ur terminal cd into ur react app and type "npm start" 



For the Back-end:
Take the app.py I made and install flask and flask-cors. 
You may need to create a virtual enviornment to install flask and flask-cors(since flask and flask-cors aren't on homebrew). To do this type:
python3 -m venv path/to/venv
source path/to/venv/bin/activate

and then install the 2 libraries. 

to run the back-end type python3 app.py. 

***IN THE MESSAGE AFTER U RUN THE BACKEND IT TELLS U WHERE IT IS RUNNING*** 

it will looks smt like this:
venv) zachhixson@Zachs-MacBook-Pro snake-back-end % python3 app.py                  
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://123.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 107-232-067


***TAKE THIS ADDRESS(IT WILL BE DIF FOR U) http://123.0.0.1:5000 AND REPLACE THE IP_and_PORT VARIABLE ON THE FRONT END(THIS VARIABLE IS IN RESTART AND IN SNAKEGAME***
