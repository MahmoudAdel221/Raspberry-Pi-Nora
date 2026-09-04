The Rasspberry Pi ADMIN password is :  AUT-SMART GLASSES
___________________________________________________________________________________________________________________________
#The project folder is "Nora"
‎
‎#The python environments are 2:
‎
‎-One is in the folder named 'SG/X' And this one contains the opencv liberary which are headless (no camera preview)
‎
‎-The second one is in the folder named 'N' one contains the normal opencv liberary that support camera preview

-- to activate the 1st enviroment use "source /home/norasamrtglasses/SG/X/bin/activate" in the terminal 
-- to activate the 2nd enviroment use "source /home/norasamrtglasses/N/bin/activate" in the terminal

-- to run the project u need to open the project folder in the terminal using cd command "cd /home/norasamrtglasses/Nora"
   then u will need to type "python3 Nora.py" to run the system code.


___________________________________________________________________________________________________________________________

#To use another domain form ngrok u will need to authrize/login with ur own ngrok account to use ur custom domain u already have on ngrok
-- to login to ngrok type in terminal "ngrok config add-authtoken TOKEN" and replace the 'TOKEN' by ur own token from ur ngrok account
-- to test ur domain use "ngrok http 5000 --domain=YOUR-DOMAIN" but make sure that u running the local host on 5000 port
-- to kill the local host that are running on port 500 use "sudo fuser -k 5000/tcp"
-- to check ur local host open "http://localhost:5000" in the browser

___________________________________________________________________________________________________________________________

#rasspberry pi connect email and passowrd:
--"norasamrtglasses@gmail.com"
--"AUT-SMART GLASSES"

___________________________________________________________________________________________________________________________



