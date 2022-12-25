
import os

import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "model_builder.settings")

django.setup()
from django_q.tasks import async_task, result
# create the task
async_task('math.copysign', 2, -2)

# or with import and storing the id
from math import copysign
task_id = async_task(copysign, 2, -2)

# get the result
task_result = result(task_id)

# result returns None if the task has not been executed yet
# you can wait for it
task_result = result(task_id, 200)
print(task)
# but in most cases you will want to use a hook:

async_task('math.modf', 2.5, hook='hooks.print_result')

# hooks.py
def print_result(task):
    print(task.result)

#%%
import requests
def send_simple_message():
	return requests.post(
		"https://api.mailgun.net/v3/sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org/messages",
		auth=("api", "d50d2879271236521803ff3a34796f65-eb38c18d-b95c3514"),
		# data={"from": "Mailgun Sandbox <postmaster@sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org>",
		data={"from": "Ashley Cherian <postmaster@sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org>",
			"to": "Ashley Cherian <ashley.cherian@aptivaa.com>",
			"subject": "Hello Ashley Cherian",
			"text": "Congratulations Ashley Cherian, you just sent an email with Mailgun!  You are truly awesome!"})

send_simple_message()



send_simple_message()

# %%


def send_simple_message():
	return requests.post(
		"https://api.mailgun.net/v3/sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org/messages",
		auth=("api", "YOUR_API_KEY"),
		data={"from": "Excited User <mailgun@YOUR_DOMAIN_NAME>",
			"to": ["bar@example.com", "YOU@YOUR_DOMAIN_NAME"],
			"subject": "Hello",
			"text": "Testing some Mailgun awesomness!"})