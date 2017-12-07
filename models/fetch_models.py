import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":
    files = {
    "outline2yellow_generator_gan.pth": "12nYSYFDKW7Pk1Uf7NT50gDU73TBN3nGH",
    "outline2yellow_discriminator.pth": "1-d8tioxfYfcswmw-R_Jcg4957eEGFpXN",
    "outline2yellow_generator.pth": "1M7H8ubLPp-2r3pAwWT_euipKW4HLUIMH"
    }

    for destination, file_id in files.items():
        print(f"==> Downloading {destination}...")
        download_file_from_google_drive(file_id, destination)

    print("==> All done!")
