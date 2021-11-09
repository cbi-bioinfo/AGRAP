
var mysql = require('mysql');
var db_info = {
    "user": "root",
    "password": "cbibioinfo2019",
    "database": "rda",
    "host": "203.252.206.118",
    "port": 6655,
}

module.exports = {
    init: function () {
        return mysql.createConnection(db_info);
    },
    connect: function(conn) {
        conn.connect(function(err) {
            if(err) console.error('mysql connection error : ' + err);
            else console.log('mysql is connected successfully!');
        });
    }
}