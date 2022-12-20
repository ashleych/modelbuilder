"""model_builder URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
        path('logistic_build/', include('logistic_build.urls')),
    path('admin/', admin.site.urls),
    
]
admin.site.site_header = 'Model Builder Administration'                    # default: "Django Administration"
admin.site.index_title = 'Model Builder'                 # default: "Site administration"
admin.site.site_title = 'Model builder site admin' # default: "Django site admin"


urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
    path('__debug__/', include('debug_toolbar.urls')),
]