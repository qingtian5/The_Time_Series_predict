setwd('C:/Users/Qingt/Desktop/stock1')

IDXCOMPO_1=read.delim("TRD_Dalyr.txt",header=T)
attach(IDXCOMPO_1)
Trddt=as.Date(Trddt,format="%Y-%m-%d")

training_set = IDXCOMPO_1[1:1184, ]  
test_set = IDXCOMPO_1[1185:1214, ]

# 原始时间序列图
pdf("fig/time_series.pdf")
plot(Trddt[1:1184],training_set[,3],
     xlab='time',
     ylab='close',type='l',lwd=2,col=4)
lines(Trddt[1185:1214],test_set[,3],
       xlab='time',
       ylab='close',type='l',lwd=2,col=2)
title('Close price')
legend("topleft",c('train','test'),fill=c(4,2))
dev.off()

temp = training_set$Clsprc
# First-order diff
dtemp1 = diff(temp)
# 一阶差分图
pdf("fig/First_order_diff.pdf")
plot(Trddt[2:1184],dtemp1,
     xlab='time',
     ylab='diff_1',type='l',lwd=2,col=4)
title('First_order_diff')
dev.off()

# Second-order diff
dtemp2 = diff(dtemp1)
# 二阶差分图
pdf("fig/Second_order_diff.pdf")
plot(Trddt[3:1184],dtemp2,
     xlab='time',
     ylab='diff_2',type='l',lwd=2,col=4)
title('Second_order_diff')
dev.off()

# 画原始数据的ACF和PACF
pdf("fig/ACF_and_PACF1.pdf")
par(mfcol=c(2,1))
acf(temp,lag=160)
pacf(temp,lag=20)
dev.off()

# 画一阶差分的ACF和PACF
pdf("fig/ACF_and_PACF_of_diff1.pdf")
par(mfcol=c(2,1))
acf(dtemp1,lag=160)
pacf(dtemp1,lag=160)
dev.off()

# ARIMA
model = arima(training_set['Clsprc'], order=c(3, 1, 0),include.mean=F)
model
pdf("fig/time_series_diagnostics.pdf")
tsdiag(model,gof=20)
dev.off()

history = training_set['Clsprc']
history = history[,1]
predictions = c()

for(t in 1:30){
  model1 = arima(history, order=c(3, 1, 0),include.mean=F)
  yhat = predict(model1,1)
  # model_fit = model1.fit()
  # yhat = model_fit.forecast()
  # yhat = np.float(yhat[0])
  predictions = append(predictions,yhat$pred,after=length(predictions))
  obs = test_set[t, 3]
  history = append(history,obs,after=length(history))
}

# 预测图
par(mfcol=c(1,1))
pdf("fig/Stock_Price_Prediction.pdf")
plot(Trddt[1185:1214],test_set[,3],
     xlab='time',
     ylab='close',type='l',lwd=2,col=4)
lines(Trddt[1185:1214],predictions,
      xlab='time',
      ylab='close',type='l',lwd=2,col=2)
title('ARIMA: Stock Price Prediction')
legend("topleft",c('Stock Price','Predicted Stock Price'),fill=c(4,2))
dev.off()



model2 = arima(Clsprc, order=c(3, 1, 0),include.mean=F)
model2
residuals = model2$residuals
pdf("fig/Residuals.pdf")
par(mfcol=c(1,2))
plot(Trddt[1:1214],residuals,
     xlab='time',
     ylab='residuals',type='l',lwd=2,col=4)
title("Residuals")
density=density(residuals)
plot(density$y,
     xlab='',xaxt='n',
     ylab='density',yaxt='n',type='l',lwd=2,col=4)
title("density")
dev.off()

real=test_set['Clsprc']
real=real[,1]

rmse = sqrt(sum((real-predictions)^2)/length(real))
rmse

library(fUnitRoots)
adf1 = adfTest(temp)
adf1
adf2 = adfTest(dtemp1)
adf2

dpred = diff(predictions)

pdf("fig/DiffFit.pdf")
par(mfcol=c(1,1))
plot(Trddt[2:1184],dtemp1,
     xlab='time',
     ylab='diff_1',type='l',lwd=2,col=4)
lines(data.index[1185:1213],dpred,
      xlab='time',
      ylab='diff_1',type='l',lwd=2,col=2)
title('DiffFit')
legend("topright",c('diff_1','prediction_diff_1'),fill=c(4,2))
dev.off()




history = training_set['Clsprc']
history = history[,1]
predictions = c()

for(t in 1:30){
  model1 = arima(history, order=c(3, 1, 0),include.mean=F)
  yhat = predict(model1,1)
  # model_fit = model1.fit()
  # yhat = model_fit.forecast()
  # yhat = np.float(yhat[0])
  predictions = append(predictions,yhat$pred,after=length(predictions))
  obs = yhat$pred
  history = append(history,obs,after=length(history))
}
history[(length(history)-30):length(history)]
# 预测图
par(mfcol=c(1,1))
# pdf("fig/Stock_Price_Prediction2.pdf")
plot(Trddt[1185:1214],test_set[,3],
     xlab='time',
     ylab='close',type='l',lwd=2,col=4)
lines(Trddt[1185:1214],predictions,
      xlab='time',
      ylab='close',type='l',lwd=2,col=2)
title('ARIMA: Stock Price Prediction')
legend("topleft",c('Stock Price','Predicted Stock Price'),fill=c(4,2))
# dev.off()



