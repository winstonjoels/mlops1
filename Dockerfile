FROM python:3.7

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /d_app
ADD . /d_app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 
EXPOSE 8080

# Run the application:
CMD ["gunicorn", "d_app:d_app"]
# , "--config=config.py"
