const express = require('express');
const router = express.Router();

var db_config = require('../config/database.js');
var conn = db_config.init();
db_config.connect(conn);

//rice테이블 col 추출
var col=[];
conn.query('SHOW COLUMNS FROM rice_data', function(err, rows, fields){ 
    for( data of rows){
        col=[...col, data.Field]
    }
});

/*벼 유전자원 데이터 조회화면 */
router.get('/list/:page', function (req, res) {
    const page = req.params.page;
    const sql = `SELECT * FROM rice_data LIMIT ${(page-1)*30} , ${(page)*30}`;    
    var length;
    conn.query('SELECT COUNT(*) AS count FROM rice_data', function(err, rows, fields){ 
        length = rows[0].count;
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data.ejs', { path:"/list", type:"rice", method:"get", data : result, col: col, page:page, length: length-1});
        }                
    });
});

/*벼 유전자원 데이터 검색 초기화면 */
router.get('/search/:page', function (req, res) {
    const page = req.params.page;
    const sql = `SELECT * FROM rice_data LIMIT ${(page-1)*30} , ${(page)*30}`;    
    var length;
    conn.query('SELECT COUNT(*) AS count FROM rice_data', function(err, rows, fields){ 
        length = rows[0].count;
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-search.ejs', { path:"/search", type:"rice", method:"get",data : result, col: col, page:page, length: length-1});
        }                
    });
});

/*벼 유전자원 데이터 검색 결과화면 */
router.post('/searchResult/:page', function (req, res) {
    const page = req.params.page;
    const searchCol=req.body;
    var sql = `SELECT * FROM rice_data WHERE ${searchCol.col} = "${searchCol.search}" LIMIT ${(page-1)*30} , ${(page)*30}`;    
    var length;
    conn.query(`SELECT COUNT(*) AS count FROM rice_data WHERE ${searchCol.col} = "${searchCol.search}"`, function(err, rows, fields){ 
        length = rows[0].count;
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-search.ejs', { path:"/searchResult", type:"rice", method:"post", data : result, col: col, page:page, length: length-1, search:{col:searchCol.col, search:searchCol.search}});
        }                
    });

});

/*벼 유전자원 데이터 다중검색 초기화면 */
router.get('/advSearch/:page', function (req, res) {
    const page = req.params.page;
    const sql = `SELECT * FROM rice_data LIMIT ${(page-1)*30} , ${(page)*30}`;    
    var length;
    conn.query('SELECT COUNT(*) AS count FROM rice_data', function(err, rows, fields){ 
        length = rows[0].count;
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-multipleSearch.ejs', { path:"/advSearch",type:"rice",  method:"get",data : result, col: col, page:page, length: length-1});
        }                
    });
});

/*벼 유전자원 데이터 다중검색 결과화면 */
router.post('/advSearchResult/:page', function (req, res) {
    const page = req.params.page;
    var sql = `SELECT * FROM rice_data WHERE `;   
    const searchCol=req.body;
    if(searchCol.col!="select column"){
        sql=sql+`${searchCol.col} = "${searchCol.search}"`;} 
    if(searchCol.col2!="select column"){
        sql=sql+` AND ${searchCol.col2} = "${searchCol.search2}"`;
    } 
    if(searchCol.col3!="select column"){
        sql=sql+` AND ${searchCol.col3} = "${searchCol.search3}"`;
    }
    var length;
    conn.query(sql, function(err, rows, fields){ 
        length = rows?rows[0].count:0;
    });
    conn.query(sql+`LIMIT ${(page-1)*30} , ${(page)*30}`, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-multipleSearch.ejs', { 
                path:"/advancedSearch",
                type:"rice", 
                method:"post", 
                data : result,
                col: col, page:page, 
                length: length-1, 
                search:{
                    col:searchCol.col, 
                    search:searchCol.search,
                    col2:searchCol.col2, 
                    search2:searchCol.search2,
                    col3:searchCol.col3, 
                    search3:searchCol.search3,
                }});
        }                
    });

});

/*벼 유전자원 데이터 cloumn 초기화면 */
router.get('/column', function (req, res) {
    res.render('data-featureSearch.ejs', { path:"/column", type:"rice", method:"get", col: col})
});

/*벼 유전자원 데이터 cloumn 결과화면 */
router.post('/column', function (req, res) {
    var selCol = req.body.col;
    var sql = `SELECT ${selCol} , COUNT(${selCol}) AS cnt FROM rice_data GROUP BY ${selCol} order by cnt desc`;  
    var colGroup=[]
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            for( data of result){
                colGroup=[...colGroup, {col : data[selCol], cnt: data.cnt}]
            }
            res.render('data-featureSearch.ejs', { path:"/column", type:"rice", method:"post", colGroup : colGroup, selCol: selCol, col: col});
        }                
    });
   
});

/*벼 유전자원 데이터 cloumn 검색 결과화면 */
router.post('/columnResult/:page', function (req, res) {
    var page = req.params.page;
    var searchCol=req.body;
    var sql = `SELECT * FROM rice_data WHERE ${searchCol.col} = "${searchCol.search}" LIMIT ${(page-1)*30} , ${(page)*30}`;   
    var sql_col = `SELECT ${searchCol.col} , COUNT(${searchCol.col}) AS cnt FROM rice_data GROUP BY ${searchCol.col} order by cnt desc`;  
    var colGroup=[]
    var length;
    conn.query(`SELECT COUNT(*) AS count FROM rice_data WHERE ${searchCol.col} = "${searchCol.search}"`, function(err, rows, fields){ 
        length = rows[0].count;
    });
    conn.query(sql_col, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            for( data of result){
                colGroup=[...colGroup, {col : data[searchCol.col], cnt: data.cnt}]
            }
        }                
    });
    conn.query(sql, function(err, result, field){
        if(err){
            console.log(err);
            res.status(500).send('Internal Server  Error');
        }else{
            res.render('data-featureSearch.ejs', { 
                path:"/columnResult",
                type:"rice",
                method:"post", 
                data : result,
                col: col, 
                colGroup : colGroup,
                page:page, 
                length: length-1, 
                selCol: searchCol.col,
                search:{
                    col:searchCol.col, 
                    search:searchCol.search,
                }});
        }                
    }); 

});
module.exports = router;