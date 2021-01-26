from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import and_, or_, not_, ForeignKey
from db import db_init
from PIL import Image
import time
# instantiate the app as a flask app
app= Flask(__name__)

# tell the app where the database is
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main_db.db'
app.config['SQLALCHEMY_BINDS'] = {
    'users': 'sqlite:///users.db',
    'images': 'sqlite:///images.db'
}
# creating the database model
db = SQLAlchemy(app)
# db_init(app)

# creating login table class, pass in db model
class Users(db.Model):
    __bind_key__ = 'users'
    # primary id key for each user
    id = db.Column(db.Integer, primary_key=True)
    # username and passwords, required=True for both, necessary fields
    username = db.Column(db.String(200), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=True)
    # date_added = db.Column(db.String(10))

    def __repr__(self):
        return '<User %r>' % self.id

class Images(db.Model):
    __bind_key__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey(Users.id), nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    date_time_added = db.Column(db.String(10))

    def __repr__(self):
        return '<Image %r>' % self.id




@app.route('/', methods=['POST', 'GET'])
# render the index.html file at this route
def index():

    # if page request is method post
    if request.method=='POST':

        # retrieve the inputs from the page request
        new_username = request.form['username']
        new_password = request.form['password']
        new_email = request.form['email']

        # create a new user
        new_user = Users(username=new_username, password=new_password, email=new_email)

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

@app.route('/delete/<int:id>')
def delete(id):

    # find the user by id
    user_to_delete= Users.query.get_or_404(id)

    try:
        # delete the user
        db.session.delete(user_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'problem deleting user '

@app.route('/update/<int:id>', methods=['POST', 'GET'])
def update(id):

    # find the user by id query
    user_to_update = Users.query.get_or_404(id)

    if request.method == 'POST':
        
        # change the usernames and passwords to new inputs
        user_to_update.username = request.form['username']
        user_to_update.password = request.form['password']
    
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
        Users.password.like(request.form['password'])

    ).first()

        if check_for_user != None:
            # return str()
            # return str(check_for_user)
            return redirect('/user/' + str(check_for_user.id))
        else:
            return redirect('/login')
    else:
        return render_template('login.html')

@app.route('/user/<int:userid>')
def user(userid):
    return render_template('user.html', user=Users.query.get(userid))

@app.route('/show_image')
def show_image():
    return render_template('show_image.html')

@app.route('/upload/<int:userid>',  methods=['POST', 'GET'])
def upload(userid):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'didnt work'
        file = request.files['file']
        if file.filename == '':
            return 'no file selected'
        if file:

            img = Image.open(file)
            path = ''
            if userid == 0:
                path = '/static/guestimage.png'
                img_to_delete = Images.query.filter(
                    Images.user_id.like(0)).all()
                print(img_to_delete)
                # for img in img_to_delete:
                #         db.session.delete(img)
                #         db.session.commit()
                # if len(img_to_delete) == 1:
                #     db.session.delete(img_to_delete)
                #     db.session.commit()

                # elif len(img_to_delete) > 1:
                #     for img in img_to_delete:
                #         db.session.delete(img)
                #         db.session.commit()
            else:
                path = '/static/' + str(userid) + 'image.png'

            img.save('.' + path)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_image = Images(user_id=userid, image_path=path, date_time_added=now)

            db.session.add(new_image)
            db.session.commit()

            return render_template('show_image.html', 
                user=Users.query.get(userid), 
                url=str('..' + path)
                )

    return render_template('/upload.html', userid=userid)


# if an error, use inbuilt error debugging tool
if __name__ == '__main__':
    app.run(debug=True)