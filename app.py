from flask import Flask, render_template, url_for, request, redirect

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, or_, not_, ForeignKey
from db import db_init

from PIL import Image

import time

from pynput.keyboard import Key, Controller

from flask_mail import Mail, Message

from model import run_model

# instantiate the app as a flask app
app = Flask(__name__)

# tell the app where the main database is
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main_db.db'

# main db is split into users and images. The main_db maps them together for us
app.config['SQLALCHEMY_BINDS'] = {
    'users': 'sqlite:///users.db',
    'images': 'sqlite:///images.db'
}
# creating the database model
db = SQLAlchemy(app)
# db_init(app)

# configurations for sending email
app.config['DEBUG'] = True
app.config['TESTING'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
# app.config['MAIL_DEBUG']=True
app.config['MAIL_USERNAME'] = 'sallsdagrate@gmail.com'
app.config['MAIL_PASSWORD'] = 'Sonu11sonu'
app.config['MAIL_DEFAULT_SENDER'] = 'sallsdagrate@gmail.com'
app.config['MAIL_MAX_EMAILS'] = None
# app.config['MAIL_SUPPRESS_SEND']=False
app.config['MAIL_ASCII_ATTACHMENTS'] = False

# instantiate the mail sender
mail = Mail(app)


# creating login table class, pass in db model
class Users(db.Model):

    # binded to the users section of the overall database
    __bind_key__ = 'users'
    # primary id key for each user
    id = db.Column(db.Integer, primary_key=True)

    # username and passwords, required=True for both, necessary fields
    username = db.Column(db.String(200), nullable=False)
    password = db.Column(db.String(200), nullable=False)

    # other attributes
    email = db.Column(db.String(200), nullable=True)
    pfp_path = db.Column(db.String(200), nullable=True)

    # date_added = db.Column(db.String(10))

    def __repr__(self):
        return '<User %r>' % self.id


class Images(db.Model):

    # binded to the images part of the overall db
    __bind_key__ = 'images'
    # primary id key for each image
    id = db.Column(db.Integer, primary_key=True)
    # foreign key user_id, binded to 'id' in the Users table
    user_id = db.Column(db.Integer, ForeignKey(Users.id), nullable=True)

    # other attributes
    image_path = db.Column(db.String(200), nullable=False)
    cancer_class = db.Column(db.Integer, nullable=True)
    scan_image_path = db.Column(db.String(200), nullable=True)
    notes = db.Column(db.String(200), nullable=True)
    date_time_added = db.Column(db.String(10))

    def __repr__(self):
        return '<Image %r>' % self.id


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    # r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    # r.headers["Pragma"] = "no-cache"
    # r.headers["Expires"] = "0"
    # r.headers['Cache-Control'] = 'public, max-age=0'
    r.cache_control.max_age = 0

    print('no cache')
    return r


@app.route('/', methods=['POST', 'GET'])
# render the index.html file at this route
def index():

    # if page request is method post
    if request.method == 'POST':

        # retrieve the inputs from the page request
        new_username = request.form['username']
        new_password = request.form['password']
        new_email = request.form['email']

        # create a new user
        new_user = Users(username=new_username,
                         password=new_password,
                         email=new_email,
                         pfp_path=None)
        try:
            # try adding to the database
            db.session.add(new_user)
            # commit to the database
            db.session.commit()
            # redirect back to the index page
            return redirect('/')
        except:
            return 'there was an error adding ur new account'
    else:

        # if not a post method, retrieve all the current users
        users = Users.query.all()
        images = Images.query.all()
        # render index.html
        return render_template('index.html', users=users, images=images)


@app.route('/user_home')
def user_home():
    return render_template('user_home.html')


@app.route('/delete/<int:id>')
def delete(id):

    # find the user by id
    user_to_delete = Users.query.get_or_404(id)
    images_to_delete = Images.query.filter(Images.user_id.like(id)).all()
    # print(images_to_delete)
    # print(len(images_to_delete))
    # try:
    # delete the user
    db.session.delete(user_to_delete)
    # db.session.delete(images_to_delete)
    if len(images_to_delete) == 0:
        pass
    else:
        for image in images_to_delete:
            db.session.delete(image)

    db.session.commit()
    return redirect('/')
    # except:
    #     return 'problem deleting user '


@app.route('/update/<int:id>', methods=['POST', 'GET'])
def update(id):

    # find the user by id query
    user_to_update = Users.query.get_or_404(id)

    if request.method == 'POST':

        # change the usernames and passwords to new inputs
        user_to_update.username = request.form['username']
        user_to_update.password = request.form['password']
        user_to_update.email = request.form['email']
        try:
            # commit and redirect to user page signed in
            db.session.commit()
            return redirect('/user/' + str(int(id)))
        except:
            return 'problem updating user'

    else:
        return render_template('update.html', user=user_to_update)


@app.route('/login', methods=['POST', 'GET'])
def login():

    if request.method == 'POST':
        check_for_user = Users.query.filter(
            Users.username.like(request.form['username']),
            Users.password.like(request.form['password'])).first()

        if check_for_user != None:
            # return str()
            # return str(check_for_user)
            return redirect('/user/' + str(check_for_user.id))
        else:
            return redirect('/login')
    else:
        return render_template('login.html')


@app.route('/create_account')
def create_account():
    return render_template('create_account.html')


@app.route('/user/<int:userid>')
def user(userid):
    # if ther user is a guest
    if userid == 0:
        # images is a list with one item.
        # the item is a return from an sql request where the one filter is
        # userid. The results are ranked by date and time and only the
        # the first one is picked.
        images = [
            Images.query.filter(Images.user_id.like(userid)).order_by(
                Images.date_time_added.desc()).first()
        ]
    else:
        # otherwise, we know there may be mulitple images
        # we just request for all of the images under that userid
        # and return them all.
        images = Images.query.filter(Images.user_id.like(userid)).order_by(
            Images.date_time_added.desc()).all()
    # pass images and user into the html render
    return render_template('user.html',
                           user=Users.query.get(userid),
                           images=images)


def reload():
    keyboard = Controller()

    keyboard.press(Key.cmd)
    keyboard.press(Key.shift)
    keyboard.press('r')

    keyboard.release(Key.cmd)
    keyboard.release(Key.shift)
    keyboard.release('r')


@app.route('/show_image/<int:userid>/<timestamp>')
def show_image(userid, timestamp):
    # return reloaded
    now = timestamp
    images = Images.query.filter(Images.date_time_added.like(now)).all()
    print(images)
    return render_template('show_image.html', userid=userid, images=images)


@app.route('/upload/<int:userid>', methods=['POST', 'GET'])
# we only need to know the user's id in order to save an image
def upload(userid):

    # if the page is sending a post request
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file[]' not in request.files:
            # if there is no file at all then return this
            return 'didnt work'
        # otherwise retrieve the file
        file = request.files.getlist('file[]')
        print(file)
        # if the file name is blank then basically no file was selected.
        # return back to the same page
        # if file.filename == '':
        #     return redirect('/upload/' + str(userid))

        # if the file path is satisfactory then

        # get the current date and time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for image in file:
            print(image)
            if file:
                # open the image using pillow
                img = Image.open(image)
                # find its dimensions
                width, height = img.size
                # if the file is not standard size
                if img.size != (512, 512):
                    print(img.size)
                    # then either resixe the image or...
                    # img = img.resize((512, 512))

                    # Crop the center of the image
                    new_width, new_height = (512, 512)
                    left = (width - new_width) / 2
                    top = (height - new_height) / 2
                    right = (width + new_width) / 2
                    bottom = (height + new_height) / 2

                    img = img.crop((left, top, right, bottom))

                # if the image is too small then say it is too small
                elif width < 256 or height < 256:
                    # we will deal with this later
                    print(img.size, 'too smol')

                # initialise file path to nothing
                path = ''
                # if the user is signed in as a guest
                if userid == 0:
                    # use the standard file path, overwrite the original if it exists.
                    path = '/static/images/guestimage.png'
                else:
                    # otherwise if the user is signed in

                    # count how many images the user has already entered
                    # these should be saves seperately and not overwritten
                    images = Images.query.filter(
                        Images.user_id.like(userid)).all()
                    count = len(images)

                    # create a unique file name based on the user and how many images they have already entered.
                    # this allows us to recreate the path easily in a way that is user specific
                    path = '/static/images/%s_%s_image.png' % (str(userid),
                                                               count)

                # save the image
                img.save('.' + path)

                output = run_model('.' + path)
                print(output)

                # create a new image object to store
                new_image = Images(
                    user_id=userid,
                    image_path=path,
                    # default scan image and notes to nothing for now
                    scan_image_path='',
                    cancer_class=int(output['cancer_class']),
                    notes=str(output),
                    date_time_added=now)

                # add the image to the db
                db.session.add(new_image)
                db.session.commit()

                print(path)

            # return the show_image page to display the image the person just uploaded
            # return render_template('show_image.html',
            #     userid=str(userid),
            #     # pass in the new image path
            #     image_path=path,
            #     )

        return redirect(url_for('.show_image', userid=userid, timestamp=now))

    # if request is not post, render upload page
    return render_template('/upload.html', userid=userid)


@app.route('/view/<int:userid>/<int:scanid>')
def view(userid, scanid):
    if userid == 0:
        user = userid
    else:
        user = Users.query.get_or_404(userid)
    return render_template('view_scan.html',
                           user=user,
                           scan=Images.query.get_or_404(scanid))


@app.route('/update_scan/<int:scanid>', methods=['POST', 'GET'])
def update_scan(scanid):
    if request.method == 'POST':
        # store the notes
        scan_to_update = Images.query.get_or_404(scanid)
        scan_to_update.notes = request.form['notes']
        db.session.commit()
        return redirect('/user/%r' % scan_to_update.user_id)
    else:
        return render_template('update_scan.html',
                               scan=Images.query.get_or_404(scanid))


@app.route('/send_email/<int:scanid>', methods=['POST', 'GET'])
def send_email(scanid):
    if request.method == 'POST':

        # gets the message and email from the form
        message = request.form['message']
        notes = request.form['notes']
        email = request.form['email']
        scan = Images.query.get_or_404(scanid)

        user = Users.query.filter(Users.id.like(scan.user_id)).first()

        # creates the message header
        # for now we will make the recipient default

        # adding the user to the title
        msg = Message('%r shared a scan with you' % user.username)

        # add the recipient
        msg.add_recipient(email)

        # we will create the email with html so we can add images and attachments
        # msg.html='<h1>this is a test email hihi</h1>'
        msg.body = str(message) + '\n\n' + str(notes)
        print('message = %r' % message)

        # add attachments
        with app.open_resource('.' + scan.image_path) as fp:
            msg.attach("scan.png", "image/png", fp.read())

        # sends the message
        mail.send(msg)
        # if we see this then the message sending was successful
        return render_template('message_sent.html', user=user)
    else:
        return render_template('send_email.html',
                               scan=Images.query.get_or_404(scanid))


# if an error, use inbuilt error debugging tool
if __name__ == '__main__':
    app.run(debug=True)

# @app.context_processor
# def override_url_for():
#     return dict(url_for=dated_url_for)

# def dated_url_for(endpoint, **values):
#     if endpoint == 'static':
#         filename = values.get('filename', None)
#         if filename:
#             file_path = os.path.join(app.root_path,
#                                  endpoint, filename)
#             values['q'] = int(os.stat(file_path).st_mtime)
#     return url_for(endpoint, **values)
