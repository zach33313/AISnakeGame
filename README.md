For the Front-end: 

Make a react app(IN TYPESCRIPT called anything(npx create-react-app snake-front-end --template typescript). 
Take the App.tsx I have in github and replace the og App.tsx. Add restart.tsx, SnakeGame.tsx, socket.tsx and VersusSnakeGame.tsx(with all associated css files) into the src folder in ur react app. 
As long as you have React and Typescript installed its chill 

 ***NEVERMIND***: You will need to install sockets using this command: npm install socket.io-client

***YOU WILL NEED TO REPLACE THE connection string(ip+port) VARIABLE IN SOCKET.TSX WITH WHERE UR BACKEND IS RUNNING*** but thats lwk the only change u need to make. 

to run the frontend go to ur terminal cd into ur react app and type "npm start" 



For the Back-end:
Take the app.py I made and install flask, flask-cors, flask-socketio and eventlet.
You may need to create a virtual enviornment to install flask and flask-cors(since flask and flask-cors aren't on homebrew). To do this type:
python3 -m venv path/to/venv
source path/to/venv/bin/activate

and then install the 4 libraries.
(pip3 install flask flask-cors flask-socketio eventlet)

to run the back-end type python3 app.py. 

***IN THE MESSAGE AFTER U RUN THE BACKEND IT TELLS U WHERE IT IS RUNNING*** 

it will looks smt like this:
venv) zachhixson@Zachs-MacBook-Pro snake-back-end % python3 app.py
(69182) wsgi starting up on http://###.#.#.#:5000 <---- this is the connection you paste into socket.tsx


***TAKE THIS ADDRESS(IT WILL BE DIF FOR U) http://123.0.0.1:5000 AND REPLACE SOCKET.TSX VARIABLE ON THE FRONT END***
