var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var db_config = require(__dirname + '/config/database.js');
var conn = db_config.init();
var bodyParser = require('body-parser');

db_config.connect(conn);

var indexRouter = require('./routes/index');
var fileProcessRouter = require('./routes/fileProcess');
var menuRouter = require('./routes/menu');
var testRouter = require('./routes/test');
var usersRouter = require('./routes/users');
var fileRouter = require('./routes/file');
var resultRouter = require('./routes/results');
var dbPepperRouter = require('./routes/pepper');
var dbRiceRouter = require('./routes/rice');
var app = express();
app.use(bodyParser.json());
// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/pepper', dbPepperRouter);
app.use('/rice', dbRiceRouter);
app.use('/file', fileProcessRouter);
app.use('/menu', menuRouter);
app.use('/users', usersRouter);
app.use('/test', testRouter);
app.use('/upload',fileRouter);
app.use('/results', resultRouter);
// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
