import random
import os

rounds=0
name=None

name=input("Please Enter Your Name :")
print("Hi "+name)
with open('result.txt' , 'a') as file:
      file.write("Player 1 : " + name +"\n")
      file.write("Player 2 : Computer" + "\n" )

print("***************** Welcome To Game of Rock , Paper , Scissor *********************")

while True:
    rounds+=1
    choices = ['rock' , 'paper' , 'scissor']
    player = None
    computerchoice=random.choice(choices)

    while player not in choices:
       choices = ['rock' , 'paper' , 'scissor']
       computerchoice=random.choice(choices)
       print("--------------------ROUND"+str(rounds)+" ----------------------------")
       player = input("Please Pick from Rock , Paper or Scissor :").lower()

    if player == computerchoice:
        with open('result.txt' , 'a') as file:
            result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'tie\n'
            print(result_string)
            file.write("-------ROUND "+str(rounds)+" ------\n")
            file.write(result_string)

    elif player=="rock":

        if computerchoice=="paper":
                with open('result.txt' , 'a') as file:
                    result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU LOOSE!!\n'
                    print(result_string)
                    file.write("-------ROUND "+str(rounds)+" ------\n")
                    file.write(result_string)

        if computerchoice=="scissor":
            with open('result.txt' , 'a') as file:
                    result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU WON!!\n'
                 
                    print(result_string)
                    file.write("-------ROUND "+str(rounds)+" ------\n")
                    file.write(result_string)

    elif player=="paper":

        if computerchoice=="scissor":
                    with open('result.txt' , 'a') as file:
                        result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU LOOSE!!\n'
                        print(result_string)
                        file.write("-------ROUND "+str(rounds)+" ------\n")
                        file.write(result_string)

        if computerchoice=="rock":
                    with open('result.txt' , 'a') as file:
                        result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU WON!!\n'
                        print(result_string)
                        file.write("-------ROUND "+str(rounds)+" ------\n")
                        file.write(result_string)

    elif player=="scissor":

        if computerchoice=="paper":
                    with open('result.txt' , 'a') as file:
                        result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU WON!!\n'
                        print(result_string)
                        file.write("-------ROUND "+str(rounds)+" ------\n")
                        file.write(result_string)

        if computerchoice=="rock":
                with open('result.txt' , 'a') as file:
                    result_string = "computer choice was: " + computerchoice + "\n" \
                        + name + " choice was: " + player + "\n" \
                        + 'YOU LOOSE!!\n'
                    print(result_string)
                    file.write("-------ROUND "+str(rounds)+" ------\n")
                    file.write(result_string)

    last_choice=input("Do You want to Play again (Yes/No)").lower()

    if last_choice!='yes':
        break
