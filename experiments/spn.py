# pip install waybackpy
from waybackpy import WaybackMachineSaveAPI

url = "https://mcdn.podbean.com/mf/web/eeh4wn5h35s8gi3r/91225_podcast8qn98.mp3"
# user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
user_agent = "Podcast Pile Archiver [https://huggingface.co/podcasts-org]"

save_api = WaybackMachineSaveAPI(url, user_agent)
archive_url = save_api.save()
print(archive_url)
