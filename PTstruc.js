
var b = document.getElementById("b");
var c = document.getElementById("c");
var ctx = c.getContext('2d');

var numPts = 500;
var goal_w = $V([Math.random()*10-5,Math.random()*10-5, Math.random()*10-5]);
 
var xyToScreen = function(pt){
	return [pt[0] + c.width/2.0, -pt[1] + c.height/2.0];
}

var generatePoints = function(num){
	var np = []
	for(var i = 0; i < num; i++){
		np.push[Math.random()*c.width];
		var xv = Math.random()*c.width;
	}
}

var xp = _.map(_.range(numPts),
	function(ind) {
		return $V([1, Math.random()*c.width - c.width/2, Math.random()*c.height - c.height/2]);
	});
var yp = _.map(xp,
	function(pt){
		if(goal_w.dot(pt)<0){
			return -1;
		}
		return 1;
	});

var solve = function(x, y){
	var xl = _.map(x, 
		function(pt){
			return pt.elements; 
		});
	var xm = $M(xl);
	var ym = $V(y);
	var pseudoInv = ((xm.transpose()).multiply(xm)).inv().multiply(xm.transpose());
	var ow = pseudoInv.multiply(ym);
	return ow;
}
	
var selectPoint = function(ind){
	if (ind==-1) {return;}
	var vpos = [xp[ind].elements[1], xp[ind].elements[2]];
	var pos = xyToScreen(vpos);
	//console.log(xp[ind].elements);
	var rad = 7;
	ctx.beginPath();
	ctx.arc(pos[0], pos[1], rad, 0, 2 * Math.PI, false);
	ctx.fillStyle = 'black';
	ctx.fill();
}
	
//x - list of vetors
//y - list of {1,-1}
var stepPLA = function(w, x, y){
	var alph = 0.5; //how intense a point has an effect
	var sz = x.length;
	var ind = Math.floor(Math.random()*sz);
	var nw = w.add(x[ind].x(y[ind]*alph));
	return [nw, ind];
}
	
var drawPLA = function(w, col) {
	var wa = w.elements;
	var inter = -wa[0]/wa[2];
	var slp = -wa[1]/wa[2];
	
	var p1 = xyToScreen([-c.width, -c.width*slp + inter]);
	var p2 = xyToScreen([c.width, c.width*slp + inter]);
	ctx.beginPath();
	ctx.moveTo(p1[0], p1[1]);
	ctx.lineTo(p2[0], p2[1]);
	ctx.strokeStyle = col;
	ctx.stroke();
}
	
var drawPoints = function() {
	//console.log('dr');
	_.each(_.range(numPts), 
	function(ind) {
		var vpos = [xp[ind].elements[1], xp[ind].elements[2]];
		var pos = xyToScreen(vpos);
		//console.log(xp[ind].elements);
		var rad = 5;
		
		ctx.beginPath();
		ctx.arc(pos[0], pos[1], rad, 0, 2 * Math.PI, false);
		if(yp[ind] > 0){
			ctx.fillStyle = 'blue';
		}else{
			ctx.fillStyle = 'red';
		}
		ctx.fill();
	});
}

var guess_w = $V([0, 1, -1]);
var solved_w = solve(xp,yp);
var auto = false;
var sel_ind = -1;

b.onclick = function() { auto = !auto };

var update = function() {
	ctx.fillStyle="#ffffff";
	ctx.fillRect(0,0,c.width,c.height);
	drawPoints();
	drawPLA(goal_w, 'green');
	drawPLA(solved_w, 'black');
	drawPLA(guess_w, 'brown');
	if (auto) {
		guess_w = stepPLA(guess_w, xp, yp)[0];
	}else{
		selectPoint(sel_ind);
	}
	window.requestAnimationFrame(update);
}

var clicked = function(e) {
	var x = e.offsetX-300;
	var y = -e.offsetY+300;
	var rt = stepPLA(guess_w, xp, yp)
	guess_w = rt[0];
	sel_ind = rt[1];
};

c.addEventListener("click",clicked);
window.requestAnimationFrame(update);
