class ToDoList:
    def __init__(self):
        self.tasks = []   # Creating a list to store tasks at each index

    def add_task(self, task):
        self.tasks.append(task)  # Keep appending tasks in the list
        print(f"Task '{task}' added.")

    def update_task(self, task_index, new_task):
        if 0 <= task_index < len(self.tasks):   # Check if user is not checking index out of bounds
            self.tasks[task_index] = new_task
            print(f"Task {task_index} updated to '{new_task}'.")
        else:
            print("Invalid task index.")

    def mark_completed(self, task_index):
        if 0 <= task_index < len(self.tasks):
            completed_task = self.tasks.pop(task_index)  # once task is completed pop it out of list
            print(f"Task '{completed_task}' marked as completed.")
        else:
            print("Invalid task index.")

    def display_tasks(self):
        if self.tasks:
            print("To-Do List:")
            for index, task in enumerate(self.tasks):   
                print(f"{index}. {task}")
        else:
            print("No tasks in the to-do list.")

def main():
    TD = ToDoList()  # Creating a object of todolist class

    while True:
        print("Hello There !! Please Select From following Options :")
        print("1. Add Task")
        print("2. Update Task")
        print("3. Mark Task as Completed")
        print("4. Display Tasks")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            task = input("Enter task: ")
            TD.add_task(task)

        elif choice == "2":
            task_index = int(input("Enter task index to update: "))
            new_task = input("Enter new task: ")
            TD.update_task(task_index, new_task)

        elif choice == "3":
            task_index = int(input("Enter task index to mark as completed: "))
            TD.mark_completed(task_index)

        elif choice == "4":
            TD.display_tasks()

        elif choice == "5":
            print("Exiting the To-Do List application. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
