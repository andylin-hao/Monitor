class ProgressBar {
    constructor(parameter) {
        this.type = parameter['type'];
        this.container = parameter['container'];
        this.background = parameter['background'];
        this.foreground = parameter['foreground'];
        this.complete = parameter['complete'];
        this.paintRegion = null;
        this.current_progress = 0;
        this.regionWidth = 200;
        this.regionHeight = 200;

        this.create_paintRegion();
        this.paint(this.current_progress);
    }

    create_paintRegion() {
        this.paintRegion = document.createElement('canvas');
        this.paintRegion.setAttribute('width', String(this.regionWidth));
        this.paintRegion.setAttribute('height', String(this.regionHeight));
        document.querySelector(this.container).appendChild(this.paintRegion);
    }

    paint(progress) {
        if (this.type === 'ring')
            this.paint_ring(this.current_progress);
        if (this.type === 'bar')
            this.paint_bar(this.current_progress);
        if (this.type === 'hourglass')
            this.paint_hourglass(this.current_progress);
    }

    paint_ring(progress) {
        let ringRadius = this.regionWidth * 0.6 / 2;
        let context = this.paintRegion.getContext('2d');
        context.lineWidth = 20;

        context.clearRect(0, 0, this.regionWidth, this.regionHeight);

        //绘制底色
        context.beginPath();
        context.strokeStyle = this.background;
        context.arc(this.regionWidth / 2, this.regionHeight / 2, ringRadius, -Math.PI / 2, 3 * Math.PI / 2, false);
        context.stroke();
        context.closePath();

        // 绘制进度条
        context.beginPath();
        context.strokeStyle = this.foreground;
        let endAngle = 2 * Math.PI * progress / 100 - Math.PI / 2;
        context.arc(this.regionWidth / 2, this.regionHeight / 2, ringRadius, -Math.PI / 2, endAngle, false);
        context.stroke();
        context.closePath();

        // 绘制文字
        context.font = "14px 微软雅黑";
        context.fillStyle = '#000000';
        context.textBaseline = 'middle';
        let text = (String(this.current_progress.toFixed(2)) + "%");
        let textWidth = context.measureText(text).width;
        context.fillText(text, this.regionWidth / 2 - textWidth / 2, this.regionHeight / 2);
    }

    paint_bar(progress) {
        let context = this.paintRegion.getContext('2d');
        let barHeight = this.regionHeight / 10;
        let radius = barHeight / 2;
        let barLength = this.regionWidth * 0.9;
        let rectLength = barLength - radius * 2;

        context.clearRect(0, 0, this.regionWidth, this.regionHeight);

        //绘制左圆底色
        context.beginPath();
        context.fillStyle = this.background;
        context.arc(this.regionWidth / 2 - rectLength / 2, this.regionHeight / 2, radius, Math.PI / 2, 3 * Math.PI / 2, false);
        context.closePath();
        context.fill();

        //绘制矩形底色
        context.fillRect(this.regionWidth / 2 - rectLength / 2, this.regionHeight / 2 - barHeight / 2, rectLength, barHeight);

        //绘制右圆底色
        context.beginPath();
        context.arc(this.regionWidth / 2 + rectLength / 2, this.regionHeight / 2, radius, -Math.PI / 2, Math.PI / 2, false);
        context.closePath();
        context.fill();


        let leftBackRadius;
        let barBackLength;
        let rightBackRadius;
        if (progress / 100 > radius / barLength) {
            leftBackRadius = radius;
            if (progress / 100 > (radius + rectLength) / barLength) {
                barBackLength = rectLength;
                rightBackRadius = radius - (progress * barLength / 100 - rectLength - radius);
            }
            else {
                barBackLength = progress * barLength / 100 - radius;
                rightBackRadius = radius;
            }
        }
        else {
            leftBackRadius = progress * barLength / 100;
            barBackLength = 0;
            rightBackRadius = radius;
        }

        //绘制左圆进度
        context.fillStyle = this.foreground;
        context.beginPath();
        let angle = Math.asin((radius - leftBackRadius) / radius);
        context.arc(this.regionWidth / 2 - rectLength / 2, this.regionHeight / 2, radius, -Math.PI / 2 - angle, Math.PI / 2 + angle, true);
        context.closePath();
        context.fill();

        //绘制中间矩形进度
        context.fillRect(this.regionWidth / 2 - rectLength / 2, this.regionHeight / 2 - barHeight / 2, barBackLength, barHeight);

        //绘制右圆进度
        angle = Math.asin((radius - rightBackRadius) / radius);
        if (angle !== 0) {
            context.beginPath();
            context.arc(this.regionWidth / 2 + rectLength / 2, this.regionHeight / 2, radius, -Math.PI / 2 + angle, -3 * Math.PI / 2 - angle, true);
            context.closePath();
            context.fill();
        }

        context.font = "14px 微软雅黑";
        context.fillStyle = '#000000';
        context.textBaseline = 'middle';
        let text = (String(this.current_progress.toFixed(0)) + "%");
        let textWidth = context.measureText(text).width;
        context.fillText(text, this.regionWidth / 2 - textWidth / 2, 7 * this.regionHeight / 10);
    }

    paint_hourglass(progress) {
        let sideLength = this.regionWidth / 10;
        let height = this.regionHeight / 10;
        let angle = Math.atan(height / (sideLength / 2));
        let context = this.paintRegion.getContext('2d');

        context.clearRect(0, 0, this.regionWidth, this.regionHeight);

        //绘制上三角底色
        context.fillStyle = this.background;
        context.beginPath();
        context.moveTo(this.regionWidth / 2 - sideLength / 2, this.regionHeight / 2 - height);
        context.lineTo(this.regionWidth / 2 + sideLength / 2, this.regionHeight / 2 - height);
        context.lineTo(this.regionWidth / 2, this.regionHeight / 2);
        context.lineTo(this.regionWidth / 2 - sideLength / 2, this.regionHeight / 2 - height);
        context.closePath();
        context.fill();

        //绘制下三角底色
        context.fillStyle = this.foreground;
        context.beginPath();
        context.moveTo(this.regionWidth / 2 - sideLength / 2, this.regionHeight / 2 + height);
        context.lineTo(this.regionWidth / 2 + sideLength / 2, this.regionHeight / 2 + height);
        context.lineTo(this.regionWidth / 2, this.regionHeight / 2);
        context.lineTo(this.regionWidth / 2 - sideLength / 2, this.regionHeight / 2 + height);
        context.closePath();
        context.fill();

        //绘制上三角进度
        let newHeight = height * (1 - progress / 100);
        let newSideLength = newHeight / Math.tan(angle) * 2;
        context.fillStyle = this.foreground;
        context.beginPath();
        context.moveTo(this.regionWidth / 2 - newSideLength / 2, this.regionHeight / 2 - newHeight);
        context.lineTo(this.regionWidth / 2 + newSideLength / 2, this.regionHeight / 2 - newHeight);
        context.lineTo(this.regionWidth / 2, this.regionHeight / 2);
        context.lineTo(this.regionWidth / 2 - newSideLength / 2, this.regionHeight / 2 - newHeight);
        context.closePath();
        context.fill();

        //绘制下三角进度
        context.fillStyle = this.background;
        context.beginPath();
        context.moveTo(this.regionWidth / 2 - newSideLength / 2, this.regionHeight / 2 + newHeight);
        context.lineTo(this.regionWidth / 2 + newSideLength / 2, this.regionHeight / 2 + newHeight);
        context.lineTo(this.regionWidth / 2, this.regionHeight / 2);
        context.lineTo(this.regionWidth / 2 - newSideLength / 2, this.regionHeight / 2 + newHeight);
        context.closePath();
        context.fill();

        context.font = "14px 微软雅黑";
        context.fillStyle = '#000000';
        context.textBaseline = 'middle';
        let text = (String(this.current_progress) + "%");
        let textWidth = context.measureText(text).width;
        context.fillText(text, this.regionWidth / 2 - textWidth / 2, 7 * this.regionHeight / 10);
    }

    set_progress(current) {
        this.current_progress = current;
        this.paint(this.current_progress);

        if (current === 100)
            this.complete();
    }
}