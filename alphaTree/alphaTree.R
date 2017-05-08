library(rjson) # make sure this package is installed

alpha.div <- function(c1, c2, alpha){
  # Computes $\alpha$-divergence between two distributions P and Q.
  # Args:
  #		c1: positive class ratio of P
  #		c2: positive class ratio of Q
  #		alpha: alpha parameter ranging from $-\infty$ to $\infty$
  # Returns:
  #	The $\alpha$ divergence between P and Q with the parameter $\alpha$
  
  # Initialization
  adiv <- -Inf; 
  
  # Special cases when $\alpha=1$ or $\alpha=0$
  if(alpha==1.0){ adiv <- sum(log(c1/c2)*c1); }
  else if(alpha==0.0){ adiv <- sum(log(c2/c1)*c2);	}
  else{ 
    # $\alpha$-divergence for general cases
    adiv <- (1.0-sum((c1^(alpha)) * (c2^(1-alpha))))/alpha/(1-alpha);
  }
  
  # Error handling
  if(adiv <= 0.0){ adiv <- 0.0; }
  
  return(adiv);
}

alpha.gain <- function(x, y, alpha, y.lst, min.n){
  # Computes $\alpha$-Gain for feature "x" 
  # Args:
  #		x: feature
  #		y: target
  #		alpha: alpha parameter ranging from $-\infty$ to $\infty$
  #		y.lst: unique values of "y"
  # Returns:
  #	The $\alpha$-Gain of "x"
  
  # Initialization
  smoother <- 1e-5;
  max.gain <- -Inf;
  max.value <- NA;
  x.lst <- unique(x);
  
  # If the unique splits of "x" are more than 10, then bin "x" into 30 values.
  if(length(x.lst) > 10){
    width <- (max(x)-min(x))/10;
    x.lst <- c(seq(from=min(x), to=max(x), by=width));
  }
  
  # If "x" has only two unique values, we need only one value.
  if(length(x.lst)==2){ x.lst <- x.lst[1]; }
  
  # Search all the splits, then find the maximum $\alpha$-Gain.
  for(split in x.lst){
    y.left <- y[which(x <= split)];
    y.right <- y[which(x > split)];
    n.y <- length(y);
    n.y.left <- length(y.left);
    n.y.right <- length(y.right);
    
    if(n.y.left < min.n || n.y.right < min.n) next;
    
    n.pos <- sum(y==y.lst[1]);
    n.pos.left <- sum(y.left==y.lst[1]);
    n.pos.right <- sum(y.right==y.lst[1]);
    node.0 <- c((n.y-n.pos),n.pos);
    node.l <- c((n.y.left - n.pos.left),n.pos.left);
    node.r <- c((n.y.right - n.pos.right),n.pos.right);
    c.0 <- node.0+smoother;
    c.l <- node.l+smoother;
    c.r <- node.r+smoother;
    c.0 <- c.0/sum(c.0);
    c.l <- c.l/sum(c.l);
    c.r <- c.r/sum(c.r);
    p.l <- length(y.left)/length(y);
    p.r <- length(y.right)/length(y);
    if(length(y.left) < 3 | length(y.right) < 3 ) next;
    gain <- p.r*alpha.div(c.r, c.0, alpha) + p.l*alpha.div(c.l, c.0, alpha);
    if(gain > max.gain){
      max.gain <- gain;
      max.value <- split;
    }
  }
  return(list(value=max.value, gain=max.gain));
}


split.variable <- function(X, y, alpha, y.lst, thr, min.n){
  # Find a split variable by searching all the variables 
  # Args:
  #		X: feature matrix
  #		y: target vector
  #		alpha: alpha parameter ranging from $-\infty$ to $\infty$
  #		y.lst: the unique values of "y"
  # Returns:
  #	The split variable
  
  # Initialization
  M <- ncol(X);
  N <- nrow(X);
  max.gain <- -Inf;
  max.feature <- NA;
  max.value <- NA;
  
  th.hat <- class.ratio(y, y.lst) ;
  
  # If the total number of data is less than 3, then stop.
  if (N < 10 || th.hat==0 || th.hat==1)
  { return(list(gain=max.gain, feature=max.feature, value=max.value)); }
  
  if (thr > 0){ # lift criterion
    sigma <- sqrt(th.hat * (1-th.hat));
    if (th.hat - 1.96 * sigma / sqrt(N) > thr){
      return(list(gain=max.gain, feature=max.feature, value=max.value));		
    }
  }
  
  # Search all the variables, and choose the best.
  for(i in 1:M){
    if (length(unique(X[,i]))==1) next;
    ag <- alpha.gain(X[,i], y, alpha, y.lst, min.n);
    if(max.gain < ag$gain){
      max.gain <- ag$gain;
      max.feature <- i;
      max.value <- ag$value;
    }
  }
  return(list(gain=max.gain, feature=max.feature, value=max.value));
}

append.rule <- function(rule, var, value,op){
  # Append a new decsion rule to an existing rule
  # Args:
  #		rule: an existing rule
  #		var: variable
  #		variable: threshold value
  #		op: operator
  # Returns:
  #	A new rule
  
  newrule = ""
  if(rule==""){ newrule=paste(var,op,value, sep=""); }
  else{ newrule=paste(rule," & ", var, op, value, sep="");}
  return(newrule)
}

class.ratio <- function(y, y.lst){
  # Computes a class ratio
  # Args:
  #		y: a target vector
  #		y.lst: the unique values of "y"
  # Returns:
  #	A class ratio
  
  ratio <- mean(as.numeric(y==y.lst[1]));
  return(ratio)
}

atree <- function(X, y, alpha, option=list()){
  # Grows an $\alpha$-Tree
  # Args:
  #		X: feature matrix
  #		y: target
  #		alpha: alpha parameter
  # Returns:
  #	An $\alpha$-Tree
  
  max.depth <- 5;
  lift <- -1;	# default value; no lift criterion
  min.n <- 10;	
  y.lst <- sort(unique(y), decreasing=FALSE);
  if(class.ratio(y, y.lst) > 0.5){
    y.lst <- sort(unique(y), decreasing=TRUE);
  }
  minor.class <- class.ratio(y, y.lst);
  Major.class <- 1 - minor.class;	
  
  if ( "max.depth" %in% names(option)){
    max.depth <- option$max.depth;			
  }
  if ( "lift" %in% names(option)){
    lift <- option$lift;	
  }
  if ( "min.n" %in% names(option)){
    min.n <- option$min.n;
  }
  if ( "class" %in% names(option)){
    y.lst <- option$class;
    minor.class <- class.ratio(y, y.lst);
    Major.class <- 1 - minor.class;	
  }	
  
  N <- length(y); # a total number of data samples
  pid.lst <- rep(1,N); # a list of partition indeces
  
  rules <- data.frame(pid=1, rule="", minor.class=minor.class, Major.class=Major.class, size=N);
  thr <- minor.class * lift;
  
  # Create an $\alpha$-Tree.
  keepSplit <- TRUE;
  depth <- 1;
  while(keepSplit){
    depth <- depth + 1;
    n.pid <- length(unique(pid.lst));
    rules.tmp <- NULL;
    pid.lst.tmp <- rep(1,N);
    pid.new <- 1;
    
    # Go over the existing paritions and divide into new partitions.		
    for(pid.old in 1:n.pid){
      
      sub.idx <- which(pid.lst==pid.old);
      X.sub <- X[sub.idx,,drop=F];
      y.sub <- y[sub.idx];
      
      split.result <- split.variable(X.sub, y.sub, alpha, y.lst, thr, min.n);	
      feature <- split.result$feature;
      value <- split.result$value;
      
      if(is.na(feature)){ # No splitting variable
        pid.lst.tmp[sub.idx] <- pid.new;
        rules.tmp <- rbind(rules.tmp, 
                           data.frame(pid=pid.new, 
                                      rule=rules[pid.old,"rule"],
                                      minor.class=rules[pid.old,"minor.class"],
                                      Major.class=rules[pid.old,"Major.class"],
                                      size=length(sub.idx)));
        pid.new <- pid.new + 1;				
      }else{ # Split.
        r.var <- as.character(colnames(X)[feature]);
        r.value <- as.character(value);
        child1.idx <- sub.idx[which(X.sub[,feature,drop=F] <= value)];
        child2.idx <- sub.idx[which(X.sub[,feature,drop=F] > value)];
        
        pid.lst.tmp[child1.idx] <- pid.new;
        minor.class.1 <- class.ratio(y[child1.idx],y.lst);
        major.class.1 <- 1 - minor.class.1;	
        rules.tmp <- rbind(rules.tmp,
                           data.frame(pid=pid.new, 
                                      rule=append.rule(rules[pid.old,2],r.var,r.value,"<="),
                                      minor.class=minor.class.1, Major.class=major.class.1,
                                      size=length(child1.idx)));
        
        pid.new <- pid.new + 1;
        pid.lst.tmp[child2.idx] <- pid.new;
        minor.class.2 <- class.ratio(y[child2.idx],y.lst);
        major.class.2 <- 1 - minor.class.2;	
        rules.tmp <- rbind(rules.tmp,
                           data.frame(pid=pid.new, 
                                      rule=append.rule(rules[pid.old,2],r.var,r.value,">"),
                                      minor.class=minor.class.2, Major.class=major.class.2,
                                      size=length(child2.idx)));
        pid.new <- pid.new + 1;
      }
    }
    pid.lst <- pid.lst.tmp;
    rules <- rules.tmp;
    if (n.pid == length(unique(pid.lst))) keepSplit <- FALSE;
    if (depth > max.depth) keepSplit <- FALSE;
  }
  return(list(minor.class=y.lst[1], Major.class=y.lst[2], 
              rules=rules, baseline=minor.class));
}

predict <- function(obj, newdata){
  # Predicts
  # Args:
  #		newdata: feature matrix
  #		rules: existing rules
  # Returns:
  #	A prediction vector.
  rules <- obj$rules;
  df <- as.data.frame(newdata);
  n.rules <- nrow(rules);
  N <- nrow(newdata);
  minor.class <- rep(0,N);
  Major.class <- rep(0,N);
  
  for(i in 1:n.rules){
    idx <- eval(parse(text=as.character(rules[i,"rule"])),envir=df);
    minor.class[idx] <- minor.class[idx] + rules[i,"minor.class"];
    Major.class[idx] <- Major.class[idx] + rules[i,"Major.class"];
  }
  return(list(minor.class=minor.class, Major.class=Major.class))
}

construct.jsontree <- function(ruma, cid){
  tree <- list();	
  rule <- unique(ruma[,cid]);
  if (length(rule)==1){
    return(paste0("E[y]:",ruma[,cid],",  n:",ruma[,cid+1])); 	
  }else{
    lsub <- ruma[which(ruma[,cid]==rule[1]),,drop=F];
    rsub <- ruma[which(ruma[,cid]==rule[2]),,drop=F];	
    tree <- list(left=construct.jsontree(lsub, cid+1),
                 right=construct.jsontree(rsub, cid+1));
    names(tree) <- c(rule[1], rule[2]);
  }
  return(tree);
}

output.json <- function(obj, model.name="data"){
  
  index.file <- paste("d3js/",model.name,".html", sep="");
  data.file <- paste("d3js/",model.name,".js", sep="");
  
  at <- obj$rules;
  max.depth <- 0;
  for(i in 1:nrow(at)){
    elem <- strsplit(as.character(at[i,"rule"]),split=" & ")[[1]];
    if( max.depth < length(elem) ){
      max.depth <- length(elem);
    }
  }
  ruma <- matrix(NA, nrow=nrow(at), ncol=(max.depth+2));
  for(i in 1:nrow(at)){
    elem <- strsplit(as.character(at[i,"rule"]),split=" & ")[[1]];
    ruma[i,1:length(elem)] <- elem;
    info <- at[i,"minor.class"];
    ruma[i,(length(elem)+1)] <- as.character(round(info,3));
    info <- at[i,"size"];
    ruma[i,(length(elem)+2)] <- as.character(info);
  }
  tree <- construct.jsontree(ruma,1);
  fout <- file(data.file);
  writeLines(paste('var alphaTree=',toJSON(tree),';',sep=""), fout);
  close(fout);
  
  preamble <- '<head><meta http-equiv="Content-Type" content="text/html;charset=utf-8"><link type="text/css" rel="stylesheet" href="style.css"><script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/d3/3.4.1/d3.js"></script><script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/d3/2.7.4/d3.layout.min.js"></script><script type="text/javascript" src="';
  postamble <- '"></script></head><body><div id="body"><div id="footer">  Alpha Tree <div class="hint">maintained by Yubin Park (yubin.park@utexas.edu) <br> click or option-click to expand or collapse <br>  Original d3.js script from <br>http://mbostock.github.io/d3/talk/20111018/tree.html</div></div></div><script type="text/javascript" src="alphaTree.js"></script></body>'
  
  index.html <- paste(preamble, paste(model.name,'.js',sep=""),postamble, sep="");	
  fout <- file(index.file);
  writeLines(index.html, fout);
  close(fout);
  
}

ensemble <- function(obj1, obj2){
  minor.class <- obj1$minor.class;
  Major.class <- obj1$Major.class;
  baseline <- obj1$baseline;
  rules <- rbind(obj1$rules, obj2$rules);
  return(list(minor.class=minor.class, Major.class=Major.class,
              rules=rules, baseline=baseline));
}

unique.ensemble <- function(obj1, obj2){
  minor.class <- obj1$minor.class;
  Major.class <- obj1$Major.class;
  baseline <- obj1$baseline;
  rules <- unique(rbind(obj1$rules, obj2$rules));
  return(list(minor.class=minor.class, Major.class=Major.class,
              rules=rules, baseline=baseline));
}

ext.highlift <- function(obj, lift){
  thr <- obj$baseline * lift;
  rules <- obj$rules;
  n <- 0;
  cov <- 0;
  new.rules <- NULL;
  for ( i in 1:nrow(rules)){
    if (rules[i,"minor.class"] > thr){
      new.rules <- rbind(new.rules, data.frame(rules[i,]));			
    }
  }
  obj$rules <- new.rules;
  return(obj)
}

predict.leat <- function(obj, newdata){
  
  rules <- obj$rules;
  df <- as.data.frame(newdata);
  n.rules <- nrow(rules);
  N <- nrow(newdata);
  hl.class <- rep(0,N);
  
  for(i in 1:n.rules){
    idx <- eval(parse(text=as.character(rules[i,"rule"])),envir=df);
    hl.class[idx] <- 1;
  }
  return(hl.class)
}










