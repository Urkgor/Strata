Certificates (`*.crt`, PEM) put in this folder are trusted by the image (`update-ca-trust`): for a network whose proxy
re-signs HTTPS with the company's own certificate.  They are not committed (see `.gitignore`).  Leave it empty otherwise.
