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
    pfp_path = db.Column(db.String(200), nullable = True)
    # date_added = db.Column(db.String(10))

    def __repr__(self):
        return '<User %r>' % self.id

class Images(db.Model):
    __bind_key__ = 'images'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey(Users.id), nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    scan_image_path = db.Column(db.String(200), nullable=True)
    notes = db.Column(db.String(200), nullable=True)
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
        new_user = Users(
            username=new_username, 
            password=new_password, 
            email=new_email,
            pfp_path=None
            )
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
    images_to_delete= Images.query.filter(
        Images.user_id.like(id)
    ).all()
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
    if userid == 0:
        images = [Images.query.filter(
        Images.user_id.like(userid)
        ).order_by(Images.date_time_added.desc()).first()]
    else:
        images = Images.query.filter(
        Images.user_id.like(userid)
    ).all()
    return render_template('user.html', user=Users.query.get(userid), images=images)

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
            width, height = img.size
            if img.size != (512, 512):
                print(img.size)
                # img = img.resize((512, 512))
                new_width, new_height = (512, 512)
                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2

                # Crop the center of the image
                img = img.crop((left, top, right, bottom))
            elif width < 256 or height < 256:
                print(img.size, 'too smol')
                

            path = ''
            if userid == 0:
                path = '/static/images/guestimage.png'
                # img_to_delete = Images.query.filter(
                #     Images.user_id.like(0)).all()
                # print(img_to_delete)
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
                images = Images.query.filter(
                    Images.user_id.like(userid)
                ).all()
                count = len(images)
                path = '/static/images/%s_%s_image.png' % (str(userid), count)
            
            img.save('.' + path)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_image = Images(
                user_id=userid, 
                image_path=path, 
                scan_image_path='', 
                notes='',
                date_time_added=now)

            db.session.add(new_image)
            db.session.commit()
            print(path)
            return render_template('show_image.html', 
                userid=str(userid), 
                image_path=path,
                )

    return render_template('/upload.html', userid=userid)

@app.route('/view/<int:userid>/<int:scanid>')
def view(userid, scanid):
    return render_template('view_scan.html', user=Users.query.get_or_404(userid), scan=Images.query.get_or_404(scanid))

@app.route('/update_scan/<int:scanid>',  methods=['POST', 'GET'])
def update_scan(scanid):
    if request.method == 'POST':
        scan_to_update = Images.query.get_or_404(scanid)
        scan_to_update.notes = request.form['notes']
        db.session.commit()
        return redirect('/user/%r' % scan_to_update.user_id)
    else:
        return render_template('update_scan.html', scan=Images.query.get_or_404(scanid))


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