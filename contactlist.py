class Contact:
    def __init__(self, name, phone_number, email, address):
        self.name = name
        self.phone_number = phone_number
        self.email = email
        self.address = address

class ContactManager:
    def __init__(self):
        self.contacts = []

    def add_contact(self, contact):
        self.contacts.append(contact)
        print(f"Contact {contact.name} added successfully.")

    def view_contacts(self):
        if not self.contacts:
            print("No contacts found.")
        else:
            print("Contact List:")
            for contact in self.contacts:
                print(f"Name: {contact.name}, Phone: {contact.phone_number}")

    def search_contact(self, tosearch):
        results = []
        for contact in self.contacts:
            if tosearch.lower() in contact.name.lower() or tosearch in contact.phone_number:
                results.append(contact)

        if not results:
            print("No matching contacts found.")
        else:
            print("Matching Contacts:")
            for result in results:
                print(f"Name: {result.name}, Phone: {result.phone_number}")

    def update_contact(self, name, new_phone_number, new_email, new_address):
        for contact in self.contacts:
            if contact.name.lower() == name.lower():
                contact.phone_number = new_phone_number
                contact.email = new_email
                contact.address = new_address
                print(f"Contact {name} updated successfully.")
                return

        print(f"Contact with name {name} not found.")

    def delete_contact(self, name):
        for contact in self.contacts:
            if contact.name.lower() == name.lower():
                self.contacts.remove(contact)
                print(f"Contact {name} deleted successfully.")
                return

        print(f"Contact with name {name} not found.")




contact_manager = ContactManager()
while True:
    print('\n')
    print("Hello Welcome to COntact Manager :")
    print("1. Add Contact")
    print("2. View Contact List")
    print("3. Search Contact")
    print("4. Update Contact")
    print("5. Delete Contact")
    print("6. Exit")

    choice = input("Enter your choice (1-6): ")

    if choice == "1":
        name = input("Enter contact name: ")
        phone_number = input("Enter phone number: ")
        email = input("Enter email: ")
        address = input("Enter address: ")
        new_contact = Contact(name, phone_number, email, address)
        contact_manager.add_contact(new_contact)

    elif choice == "2":
        contact_manager.view_contacts()

    elif choice == "3":
        tosearch = input("Enter name or phone number to search: ")
        contact_manager.search_contact(tosearch)

    elif choice == "4":
        name = input("Enter name of the contact to update: ")
        new_phone = input("Enter new phone number: ")
        new_email = input("Enter new email: ")
        new_address = input("Enter new address: ")
        contact_manager.update_contact(name, new_phone, new_email, new_address)

    elif choice == "5":
        name = input("Enter name of the contact to delete: ")
        contact_manager.delete_contact(name)

    elif choice == "6":
        print("Exiting the Contact Management System. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")
