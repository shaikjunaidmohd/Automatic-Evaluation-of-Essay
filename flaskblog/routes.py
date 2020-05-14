import os
import secrets
import pickle
import numpy as np
from pymagnitude import *
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree  
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from copy import copy, deepcopy
from spellchecker import SpellChecker
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from flaskblog import app, db, bcrypt, mail
from flaskblog.forms import (RegistrationForm, LoginForm, UpdateAccountForm,EssayForm,
                             PostForm, RequestResetForm, ResetPasswordForm)
from flaskblog.models import User, Post , Essay
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message


model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/sub")
def sub():
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(username = current_user.username).first_or_404()
    essays = Essay.query.filter_by(student = user)\
        .order_by(Essay.date_posted.desc())\
        .paginate(page=page, per_page=5)  
    return render_template('sub.html', essays = essays)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password , acctype = form.acctype.data )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Examiner Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data , acctype = form.acctype.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Examiner Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, question = form.question.data , content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='Add Test',
                           form=form, legend='Add Test')


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)

def positive(es):
    l = es.split(".")
    positive=0
    negative=0
    neutral=0
    sid = SentimentIntensityAnalyzer()
    for i in l:
        ss = sid.polarity_scores(i)
        positive+=ss["pos"]
        negative+=ss["neg"]
        neutral+=ss["neu"]
    p=positive*100/len(l)
    #print("Positive =",p)
    return p
def negative(es):
    l = es.split(".")
    positive=0
    negative=0
    neutral=0
    sid = SentimentIntensityAnalyzer()
    for i in l:
        ss = sid.polarity_scores(i)
        positive+=ss["pos"]
        negative+=ss["neg"]
        neutral+=ss["neu"]
    n=negative*100/len(l)
    #print("neg",n)
    return n
def neutral(es):
    l = es.split(".")
    positive=0
    negative=0
    neutral=0
    sid = SentimentIntensityAnalyzer()
    for i in l:
        ss = sid.polarity_scores(i)
        positive+=ss["pos"]
        negative+=ss["neg"]
        neutral+=ss["neu"]
    neu=neutral*100/len(l)    
    #print("neutral",neu)
    return neu

def existential(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)    
    ex=0
    for i,j in pos:
        if j=='EX':
            ex+=1
    #print("Number of existential there :",ex)
    return ex

def predeterminants(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)   
    pdt=0
    for i,j in pos:
        if j=='PDT':
            pdt+=1
    #print("Number of predeterminants :",pdt)
    return pdt   

def superlat(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)   
    superlative=0
    for i,j in pos:
        if j=='JJS':
            superlative+=1
    #print("Number of superlative adjectives :",superlative) 
    return superlative    

def n3rdverb(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    n3rd_verb = 0
    for i,j in pos:
        if(j == 'VBP'):
            n3rd_verb +=1
    #print("Number of verb non 3rd person singular present :",n3rd_verb)
    return n3rd_verb

def pastptverb(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    pastp_verb = 0
    for i,j in pos:
        if(j == 'VBN'):
            pastp_verb +=1
    #print("Number of verb past participle :",pastp_verb)
    return pastp_verb

def mostfreq(es):
    tokens=word_tokenize(es)
    se  = set()
    d = {}
    repeat = 0 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in tokens:
        if i not in punctuations and i in se:
            d[i] += 1 
        else:
            se.add(i)
            d[i] = 0
    #print("Number of most frequent repetitive word :",max(d.values()))
    return  max(d.values())    

def cc(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)  
    cc_count = 0
    for i,j in pos:
        if(j == 'CC'):
            cc_count +=1
    #print("Number of coordinating-conjunctions :",cc_count)
    return cc_count
 
def pastverb(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    past_verb = 0
    for i,j in pos:
        if(j == 'VBD'):
            past_verb +=1
    #print("Number of past tense verb :",past_verb)
    return past_verb

def baseverb(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    base_verb = 0
    for i,j in pos:
        if(j == 'VB'):
            base_verb +=1
    #print("Number of base verb form :",base_verb)
    return base_verb

def ppverb(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    pp_verb = 0
    for i,j in pos:
        if(j == 'VBG'):
            pp_verb +=1
    #print("Number of verb present participle :",pp_verb)
    return pp_verb    

def words_ing(es):
    ing=re.findall(r'\b(\w+ing)\b',es)
    wordsendingwithing=len(ing)
    #print("number of words ending with -ing",wordsendingwithing)
    return wordsendingwithing

def misspelled(es):
    spell = SpellChecker()
    tokens=word_tokenize(es)
    misspelled = spell.unknown(tokens)
    miscount = 0
    for i in misspelled:
        #print(spell.correction(i),i)
        miscount += 1
    #print("Number of misspelled words :",miscount)  
    return miscount  

def singular(es): 
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    singular_count = 0
    for i,j in pos:
        if(j == 'NN' or j  == 'NNP'):
            singular_count += 1
    #print("Number of singlar subjects count :",singular_count)  
    return singular_count 

def plural(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    plural_count = 0
    for i,j in pos:
        if(j == 'NNS' or j  == 'NNPS'):
            plural_count += 1
    #print("Number of plural subjects count :",plural_count)
    return plural_count    

def stop_word(s):
    stop_words = set(stopwords.words('english')) 
    vecs = Magnitude('wiki-news-300d-1M.magnitude')  
    word_tokens = word_tokenize(s) 
    
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
    filtered_sentence = ""
    
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence += " " + w 
    return (filtered_sentence)   

def max_span_tree(es):
    k = []
    q = 0
    i = 0
    vecs = Magnitude('wiki-news-300d-1M.magnitude')  
    k = es.split(".")
    #print(k)  
    ele = []
    for i in range(len(k)):
        w = 0
        k1 = k[i]
        st1 = stop_word(k1)
        ele1 = []
        for j in range(len(k)):
            k2 = k[j]
            st2 = stop_word(k2)
            w = vecs.similarity(st1, st2) 
            ele1.append(round(w,2))
        ele.append(ele1)    
    ki2 = []
    ki = 0
    for i in range(len(ele)):
        ki1 = []
        for j in range(len(ele)):
            if(ele[i][j] == 0):
                ki = 1
            else:

                ki = 1/ele[i][j]

            ki1.append(round(ki,2))
        ki2.append(ki1)
  #print(ki2)  
    X = csr_matrix(ki2)
    Tmat1 = minimum_spanning_tree(X).toarray()
    for i in range(len(Tmat1)):
        for j in range(len(Tmat1)):
            if(Tmat1[i][j] != 0):
                Tmat1[i][j] = ele[i][j]
    max1 =0
    for i in range(len(Tmat1)):
        for j in range(len(Tmat1)):
            max1 += Tmat1[i][j] 

  #print("sum of the maximum spanning tree :", max1)    
    return max1          

def density(es):
    vecs = Magnitude('wiki-news-300d-1M.magnitude')  

    k = []
    q = 0
    i = 0

    k = es.split(".")  
    ele = []
    for i in range(len(k)):
        w = 0
        k1 = k[i]
        st1 = stop_word(k1)
        ele1 = []
        for j in range(len(k)):
            k2 = k[j]
            st2 = stop_word(k2)
            w = vecs.similarity(st1, st2) 
            ele1.append(round(w,2))
        ele.append(ele1)    
    reduced_matrix= deepcopy(ele)
    for i in range(len(reduced_matrix)):
        for j in range(len(reduced_matrix)):
            if(reduced_matrix[i][j] < 0.35):
                reduced_matrix[i][j] = 0
    reduced_edges = 0
    for i in range(len(reduced_matrix)):
        for j in range(i+1,len(reduced_matrix)):
            if(reduced_matrix[i][j] != 0):
                reduced_edges += 1
    no_of_nodes = len(ele)   
    if(no_of_nodes > 1):  
        density_red = 2*reduced_edges / (no_of_nodes * (no_of_nodes - 1))
    else:
        density_red = 2*reduced_edges / (no_of_nodes)
    #print("density of reduced graph :",density_red)

    original_edges = 0
    for i in range(len(ele)):
        for j in range(len(ele)):
            if(ele[i][j] != 0):
                original_edges += 1
    if(no_of_nodes > 1):      
        density_orig = original_edges / (no_of_nodes * (no_of_nodes - 1))
    else:
        density_orig = original_edges / (no_of_nodes)
    
    #print("density of original graph :",density_orig)
    
    density_diff = abs(density_orig - density_red)
    #print("density difference between original and reduced graph is :",density_diff) 
    return density_diff

def common_length_sen(es):
    sentence = []
    sentence = es.split(".")
    csl = 0
    for i in range(len(sentence)):
        word = sentence[i].split(" ")
        if(len(word) >= 15 and len(word) <= 30):
            csl += 1
    #print("the common length sentence :", csl)
    return csl

def unique(es):
    tokens=word_tokenize(es)
    pos=nltk.pos_tag(tokens)
    p=[]
    for i,j in pos:
        p.append(j)
    po=set(p)
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = []
    for char in po:
        if char not in punctuations:
            no_punct.append(char)
    unique=list(no_punct)
    #print("unique parts of speech used :",len(unique))
    return len(unique)

def no_of_characters(es):
    sentence = es.split(".")
    words=[]
    tokens = word_tokenize(es) 
  #print("Number of characters",len(es))
    return len(es)

def no_of_words(es):
    sentence = es.split(".")
    no_of_words = 0
    for i in range(len(sentence)):
        words = sentence[i].split(" ")
        no_of_words += len(words)
  #print("number of words",no_of_words)
    return no_of_words

def top10words(m,n,tokens):
    z={}
    for i in range(len(n)):
        if (m[i] in tokens):
            z[m[i]]=n[i] 
    res={k: v for k, v in sorted(z.items(), key=lambda item: item[1],reverse=True)}
    t=[]
    for word in res:
        t.append(word)
    top=[]
    if(len(t)>=10):
        for i in range(10):
            top.append(t[i])
    else:
        top=t
    return top        

def similarity_essayprompt(es,prompt):
    vecs = Magnitude('wiki-news-300d-1M.magnitude')  
    atoken=word_tokenize(es)
    btoken=word_tokenize(prompt)
    pos_A=nltk.pos_tag(atoken)
    pos_B=nltk.pos_tag(btoken)
    noun = ['NN','NNS','NNP','NNPS']
    verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
    documentA = ''
    documentB = ''
    atokens = []
    btokens = []
    for i,j in pos_A:
        if(j in noun or j in verb):
            documentA += ' ' + i
            atokens.append(i)
    for i,j in pos_B:
        if(j in noun or j in verb):
            documentB += ' ' + i
            btokens.append(i)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([documentA, documentB])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    a=denselist[0]
    b=denselist[1]
    f=top10words(feature_names,a,atokens)
    g=top10words(feature_names,b,btokens)
    topessay = ' '.join(f)
    topprompt = ' '.join(g)
    sim = vecs.similarity(topessay, topprompt)    
    #print("Semantic similarity between the top words of essay and prompt :",sim) 
    return sim   

def min_span_tree(es):
    k = []
    q = 0
    i = 0
    vecs = Magnitude('wiki-news-300d-1M.magnitude')  
    k = es.split(".")
    #print(k)  
    ele = []
    for i in range(len(k)):
        w = 0
        k1 = k[i]
        st1 = stop_word(k1)
        ele1 = []
        for j in range(len(k)):
            k2 = k[j]
            st2 = stop_word(k2)
            w = vecs.similarity(st1, st2) 
            ele1.append(round(w,2))
        ele.append(ele1)    
    #print(ele) 
    X = csr_matrix(ele)
    Tmat = minimum_spanning_tree(X).toarray()
    res = 0
    for i in range(len(Tmat)):
        for j in range(len(Tmat)):
            res += Tmat[i][j] 
    #print("Sum of minimum spanning tree :",res)
    return res    

@app.route("/post/<int:post_id>/end", methods=['GET', 'POST'])
def end(post_id):
    post = Post.query.get_or_404(post_id)
    form = EssayForm()
    if form.validate_on_submit():
        output = 1
        int_features=[]
    
        
        
        es = form.content.data
        prompt1 = post.content
        int_features.append(no_of_characters(es))
        int_features.append(no_of_words(es))
        int_features.append(similarity_essayprompt(es,prompt1))
        int_features.append(min_span_tree(es))
        int_features.append(max_span_tree(es))
        int_features.append(density(es))
        int_features.append(common_length_sen(es))
        int_features.append(unique(es))
        int_features.append(words_ing(es))
        int_features.append(misspelled(es))
        int_features.append(singular(es))
        int_features.append(plural(es))
        int_features.append(cc(es))
        int_features.append(pastverb(es))
        int_features.append(baseverb(es))
        int_features.append(ppverb(es))
        int_features.append(n3rdverb(es))
        int_features.append(pastptverb(es))
        int_features.append(mostfreq(es))
        int_features.append(superlat(es))
        int_features.append(existential(es))
        int_features.append(predeterminants(es))
        int_features.append(positive(es))
        int_features.append(negative(es))
        int_features.append(neutral(es))
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 0)
        essay = Essay(content = form.content.data ,score = output, student = current_user , postessay = post)
        db.session.add(essay)
        db.session.commit()
        flash('Your test has been recorded! and your score is {}'.format(output), 'success')
        return redirect(url_for('home'))
    return render_template('attempt.html',form=form, post=post, user=current_user.username)



@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.question = form.question.data
        post.content = form.content.data
        db.session.commit()
        flash('Your test has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.question.data = post.question
        form.content.data = post.content
    return render_template('create_post.html', title='Update Test',
                           form=form, legend='Update Test')


@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))


@app.route("/user/<string:username>")
def user_posts(username):
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(username=username).first_or_404()
    posts = Post.query.filter_by(author=user)\
        .order_by(Post.date_posted.desc())\
        .paginate(page=page, per_page=5)  
    return render_template('user_posts.html', posts=posts, user=user)

@app.route("/post/<string:title>/view")
def view(title):
    page = request.args.get('page', 1, type=int)
    #post = Post.query.get_or_404(post_id)
    post = Post.query.filter_by(title = title).first_or_404()
    essays = Essay.query.filter_by(postessay = post)\
        .order_by(Essay.date_posted.desc())\
        .paginate(page=page, per_page=5)
    return render_template('view.html', essays = essays, post = post)   


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)