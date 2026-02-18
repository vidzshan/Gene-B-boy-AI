from appwrite.client import Client
from appwrite.services.tables_db import TablesDB
from appwrite.id import ID

client = Client()
client.set_endpoint('https://nyc.cloud.appwrite.io/v1')
client.set_project('694deac0001b5781951e')
client.set_key('standard_bae5fecc8a3223a94c4f5c249d6c047e0e8a56b7416a42a36d8028aaf2bea668147f9b84e84bfca512e5e736bbbee341add2e00796e4f452f1045e27cee4d4558e3de6ab460ea5d9e7e35e1d661debe737983518671acb160ab9a29785cbf4f05ba9469099f53cef2e730283e7c67fcff00439bd439e5846dde9fd9d1b38d57e')

tablesDB = TablesDB(client)

todoDatabase = None
todoTable = None

def prepare_database():
  global todoDatabase
  global todoTable

  todoDatabase = tablesDB.create(
    database_id=ID.unique(),
    name='TodosDB'
  )

  todoTable = tablesDB.create_table(
    database_id=todoDatabase['$id'],
    table_id=ID.unique(),
    name='Todos'
  )

  tablesDB.create_string_column(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    key='title',
    size=255,
    required=True
  )

  tablesDB.create_string_column(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    key='description',
    size=255,
    required=False,
    default='This is a test description.'
  )

  tablesDB.create_boolean_column(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    key='isComplete',
    required=True
  )

def seed_database():
  testTodo1 = {
    'title': "Buy apples",
    'description': "At least 2KGs",
    'isComplete': True
  }

  testTodo2 = {
    'title': "Wash the apples",
    'isComplete': True
  }

  testTodo3 = {
    'title': "Cut the apples",
    'description': "Don\'t forget to pack them in a box",
    'isComplete': False
  }

  tablesDB.create_row(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    row_id=ID.unique(),
    data=testTodo1
  )

  tablesDB.create_row(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    row_id=ID.unique(),
    data=testTodo2
  )

  tablesDB.create_row(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    row_id=ID.unique(),
    data=testTodo3
  )

from appwrite.query import Query

def get_todos():
  # Retrieve rows (default limit is 25)
  todos = tablesDB.list_rows(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id']
  )
  print("Todos:")
  for todo in todos['rows']:
    print(f"Title: {todo['title']}\nDescription: {todo['description']}\nIs Todo Complete: {todo['isComplete']}\n\n")

def get_completed_todos():
  # Use queries to filter completed todos with pagination
  todos = tablesDB.list_rows(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    queries=[
      Query.equal("isComplete", True),
      Query.order_desc("$createdAt"),
      Query.limit(5)
    ]
  )
  print("Completed todos (limited to 5):")
  for todo in todos['rows']:
    print(f"Title: {todo['title']}\nDescription: {todo['description']}\nIs Todo Complete: {todo['isComplete']}\n\n")

def get_incomplete_todos():
  # Query for incomplete todos
  todos = tablesDB.list_rows(
    database_id=todoDatabase['$id'],
    table_id=todoTable['$id'],
    queries=[
      Query.equal("isComplete", False),
      Query.order_asc("title")
    ]
  )
  print("Incomplete todos (ordered by title):")
  for todo in todos['rows']:
    print(f"Title: {todo['title']}\nDescription: {todo['description']}\nIs Todo Complete: {todo['isComplete']}\n\n")

if __name__ == "__main__":
  prepare_database()
  seed_database()
  get_todos()
  get_completed_todos()
  get_incomplete_todos()
