"""
Objective: create a dog with a name, a colour, and a greeting
"""



dog1Name = "Bob"
dog1Colour = "Black"
dog1Greeting = "Woof"

dog2Name = "Orange"
dog2Colour = "Orange"
dog2Greeting = "Woof"





class Dog:
    def __init__(self, name, colour):
        # these are called properties (think of them as nouns)
        self.name = name
        self.colour = colour
        self.greeting = "Woof"
    
    # these are called methods (think of thems as nouns)
    def say_hello(self):
        print("Hello there " + self.greeting)
    
    def get_info(self):
        print(self.greeting + ", my name is " + self.name + " and I am " + self.colour)


d1 = Dog("Bob", "Black")
d2 = Dog("Bob", "Black")
d3 = Dog("Bob", "Black")
d4 = Dog("Bob", "Black")
d5 = Dog("Bob", "Black")
d6 = Dog("Bob", "Black")
d7 = Dog("Bob", "Black")

d2.name


print("\nPRINTING DOG (with OOP)")
print(d1.name)
print(d1.colour)
print(d1.greeting)

d1.say_hello()
d1.get_info()







d2 = Dog("Orange", "Orange")
d3 = Dog("Mister", "White")






# AngryDog is the child class
# Dog is the parent class
# class AngryDog(Dog):
#     def __init__(self, name, colour):
#         super().__init__(name, colour)
#         self.greeting = "I hate you"
    
#     def say_hello_angrily(self):
#         print("This is a new greeting only for angry dogs")


# ad = AngryDog("Angry", "Red")
# print("\nPRINTING ANGRY DOG")
# print(ad.name)
# print(ad.colour)
# print(ad.greeting)

# ad.say_hello()
# ad.get_info()

# ad.say_hello_angrily()