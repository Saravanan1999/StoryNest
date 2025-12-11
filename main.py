"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

from storynest import generate_bedtime_story


def main() -> None:
    user_input = input("What kind of story do you want to hear? ")
    response = generate_bedtime_story(user_input)
    print(response)


if __name__ == "__main__":
    main()