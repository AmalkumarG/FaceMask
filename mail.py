from urllib.parse import urlencode
import pycurl

crl = pycurl.Curl()
crl.setopt(crl.URL, 'https://alc-training.in/gateway.php')
data = {'email': 'joelvarghese656@gmail.com','msg':'Alert'}
pf = urlencode(data)

# Sets request method to POST,
# Content-Type header to application/x-www-form-urlencoded
# and data to send in request body.
crl.setopt(crl.POSTFIELDS, pf)
crl.perform()
crl.close()