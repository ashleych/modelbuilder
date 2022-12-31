"""
Django settings for model_builder project.

Generated by 'django-admin startproject' using Django 4.1.3.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""
from decouple import config
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-c!)vk1u1je==8%gu%*hty!4xl)@8v7pc_(i9$((ff6i!19kb*5'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'django_celery_beat',
    # 'django_celery_results',
    'debug_toolbar',
    'django_q',
    # 'view_breadcrumbs',
    # 'dynamic_breadcrumbs'
    'logistic_build',
    'django_filters',
     'crispy_forms',
]

MIDDLEWARE = [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    
]

ROOT_URLCONF = 'model_builder.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'logistic_build.context_processors.cfg_assets_root',
                
            ],
        },
    },
]

WSGI_APPLICATION = 'model_builder.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'TEST': {
            'NAME': BASE_DIR / 'testdb.sqlite3',
        },
    }
}
CELERY_RESULT_BACKEND = "django-db"
# This configures Redis as the datastore between Django + Celery
CELERY_BROKER_URL = config('CELERY_BROKER_REDIS_URL', default='redis://localhost:6379')
# if you out to use os.environ the config is:
# CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_REDIS_URL', 'redis://localhost:6379')

# this allows you to schedule items in the Django admin.
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers.DatabaseScheduler'


Q_CLUSTER = {
    'name': 'logistic_build_q',
    'workers': 4,
    'recycle': 500,
    'timeout': 60,
    'compress': True,
    'save_limit': 250,
    'queue_limit': 500,
    'cpu_affinity': 1,
    'label': 'Django Q',
    'redis': 'redis://localhost:6379'
}
# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators



AUTH_PASSWORD_VALIDATORS = [
    # {
    #     'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    # },
    # {
    #     'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    # },
    # {
    #     'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    # },
    # {
    #     'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    # },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE =  'Asia/Kolkata'
# TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ASSETS_ROOT = os.getenv('ASSETS_ROOT', '/static/assets') 
import os
# ASSETS_ROOT = os.path.join(BASE_DIR, '/logistic_build/static/logistic_build/assets') 
ASSETS_ROOT = 'static/logistic_build/assets'

LOGIN_REDIRECT_URL = 'experiment_all'
LOGOUT_REDIRECT_URL = '/accounts/login/'

INTERNAL_IPS=['127.0.0.1']
# EMAIL_BACKEND = "sendgrid_backend.SendgridBackend"

# Bottom of settings.py 
# Twilio SendGrid
# EMAIL_HOST = 'smtp.sendgrid.net'
# EMAIL_PORT = 587
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = 'apikey' # Name for all the SenGrid accounts
# EMAIL_HOST_PASSWORD = config('SENDGRID_API_KEY')
# SENDGRID_SANDBOX_MODE_IN_DEBUG =False
# SENDGRID_API_KEY=config('SENDGRID_API_KEY')

ANYMAIL = {"MAILGUN_API_KEY": "d50d2879271236521803ff3a34796f65-eb38c18d-b95c3514",
"MAILGUN_SENDER_DOMAIN": "sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org",
}

EMAIL_BACKEND = "anymail.backends.mailgun.EmailBackend"
DEFAULT_FROM_EMAIL = "postmaster@sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org"
# The email you'll be sending emails from
# DEFAULT_FROM_EMAIL = config('FROM_EMAIL', default='ashley.cherian@gmail.com')
LOGIN_REDIRECT_URL = 'success'


# ANYMAIL = {
# "MAILGUN_API_KEY": "d50d2879271236521803ff3a34796f65-eb38c18d-b95c3514",
# "MAILGUN_SENDER_DOMAIN": "sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org",
# }
# EMAIL_BACKEND = "anymail.backends.mailgun.EmailBackend"
# DEFAULT_FROM_EMAIL = "postmaster@sandbox0cdfca8a97eb49fab008d9179dcff9c7.mailgun.org"

# def send_simple_message():
# 	return requests.post(
# 		f"https://api.mailgun.net/v3/{ANYMAIL['MAILGUN_SENDER_DOMAIN']}/messages",
# 		auth=("api", ANYMAIL['MAILGUN_API_KEY']),
# 		data={"from": f"Ashley C <{DEFAULT_FROM_EMAIL}>",
# 			"to": ["ashley.cherian@aptivaa.com"],
# 			"subject": "Hello New testing",
# 			"text": "helo, Testing some Mailgun awesomness!"})

# https://medium.com/@9cv9official/sending-html-email-in-django-with-anymail-7163dc332113
#  set up any mail with mailgun account
# for free tier access,need to verify recipient account 


BREADCRUMBS_HOME_LABEL = "My new home"
CRISPY_TEMPLATE_PACK = 'bootstrap4'